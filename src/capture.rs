use base64::Engine;
use ffmpeg::util::frame::video::Video;
use ffmpeg::{format, media};
use ffmpeg_next::{self as ffmpeg};
use image::codecs::jpeg;
use image::ImageBuffer;
use std::sync::Once;

static INIT: Once = Once::new();

pub(crate) fn init() {
    INIT.call_once(|| {
        ffmpeg::init().unwrap();
    });
}
pub(crate) fn capture_base64(
    input_path: &str,
    start_ms: u32,
    duration_ms: u32,
    interval_ms: u32,
) -> anyhow::Result<Vec<String>> {
    use base64::prelude::BASE64_STANDARD;

    let mut input = format::input(&input_path)?;
    let video_stream_context = input
        .streams()
        .best(media::Type::Video)
        .ok_or(anyhow::Error::from(ffmpeg::Error::StreamNotFound))?;
    let video_stream_index = video_stream_context.index();

    let video_stream = input.stream(video_stream_index).unwrap();
    let codec_params = video_stream.parameters();
    let time_base: f64 = video_stream.time_base().into();

    let context_decoder = ffmpeg::codec::context::Context::from_parameters(codec_params)?;
    let mut decoder = context_decoder.decoder().video()?;

    let mut scaler = ffmpeg::software::scaling::context::Context::get(
        decoder.format(),
        decoder.width(),
        decoder.height(),
        format::Pixel::RGB24,
        decoder.width(),
        decoder.height(),
        ffmpeg::software::scaling::Flags::BILINEAR,
    )?;

    let start_pts = start_ms as f64 / 1000.0;
    let end_pts = (start_ms + duration_ms) as f64 / 1000.0;
    let mut next_pts = start_pts;

    let mut base64_frames = Vec::new();
    let mut process_decoded_frames =
        |decoder: &mut ffmpeg::decoder::Video| -> Result<(), anyhow::Error> {
            let mut decoded = Video::empty();
            while decoder.receive_frame(&mut decoded).is_ok() {
                let mut rgb_frame = Video::empty();
                scaler.run(&decoded, &mut rgb_frame)?;
                let image_buffer = ImageBuffer::<image::Rgb<u8>, _>::from_raw(
                    rgb_frame.width(),
                    rgb_frame.height(),
                    rgb_frame.data(0).to_vec(),
                )
                .ok_or("Failed to create image buffer")
                .unwrap();

                if let Some(pts) = decoded.pts() {
                    let pts = pts as f64 * time_base;
                    if pts < start_pts || pts < next_pts {
                        continue;
                    }
                    if pts > end_pts {
                        break;
                    }
                    next_pts += interval_ms as f64 / 1000.0;
                } else {
                    return Err(anyhow::anyhow!("No pts"));
                }

                let mut jpeg_data = Vec::new();
                let mut encoder = jpeg::JpegEncoder::new_with_quality(&mut jpeg_data, 100);
                encoder.encode(
                    &image_buffer,
                    image_buffer.width(),
                    image_buffer.height(),
                    image::ExtendedColorType::Rgb8,
                )?;

                let base64_frame = BASE64_STANDARD.encode(jpeg_data);
                base64_frames.push("data:image/jpeg;base64,".to_owned() + &base64_frame);
            }
            Ok(())
        };

    for (stream, packet) in input.packets() {
        if stream.index() == video_stream_index {
            decoder.send_packet(&packet)?;
            process_decoded_frames(&mut decoder)?;
        }
    }
    decoder.send_eof()?;
    process_decoded_frames(&mut decoder)?;

    Ok(base64_frames)
}
