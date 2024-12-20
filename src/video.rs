use anyhow::Ok;
use base64::Engine;
use ffmpeg::encoder;
use ffmpeg::util::frame::{audio::Audio, video::Video};
use ffmpeg_next::{
    self as ffmpeg, channel_layout, codec, decoder, filter, format, media, picture, rescale,
    software, Dictionary, Error, Frame, Packet, Rational, Rescale,
};
use image::codecs::jpeg;
use image::ImageBuffer;
use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::Path;
use std::sync::Once;

static INIT: Once = Once::new();

pub(crate) fn init() {
    INIT.call_once(|| {
        ffmpeg::init().unwrap();
    });
}

pub(crate) fn capture_base64(
    input_path: &Path,
    start_sec: i64,
    duration_sec: i64,
    interval_msec: i64,
) -> anyhow::Result<Vec<String>> {
    use base64::prelude::BASE64_STANDARD;

    let mut input = format::input(&input_path)?;

    let start_pos = start_sec.rescale((1, 1), rescale::TIME_BASE);
    input.seek(start_pos, ..start_pos)?;

    let video_stream_context = input
        .streams()
        .best(media::Type::Video)
        .ok_or(anyhow::anyhow!(Error::StreamNotFound))?;
    let video_stream_index = video_stream_context.index();

    let video_stream = input
        .stream(video_stream_index)
        .ok_or(anyhow::anyhow!(Error::StreamNotFound))?;
    let codec_params = video_stream.parameters();
    let mut decoder = codec::context::Context::from_parameters(codec_params)?
        .decoder()
        .video()?;

    let mut scaler = software::scaling::context::Context::get(
        decoder.format(),
        decoder.width(),
        decoder.height(),
        format::Pixel::RGB24,
        decoder.width(),
        decoder.height(),
        software::scaling::Flags::BILINEAR,
    )?;

    let time_base = video_stream.time_base();
    let start_pts = start_sec.rescale((1, 1), time_base);
    let end_pts = (start_sec + duration_sec).rescale((1, 1), time_base);
    let interval = (interval_msec).rescale((1, 1000), time_base);
    let mut next_pts = start_pts;

    fs::create_dir_all("output/capture")?;
    let mut frame_count = 0;
    let mut base64_frames = Vec::new();
    let mut receive_and_process_decoded_frames =
        |decoder: &mut decoder::Video| -> Result<(), anyhow::Error> {
            let mut decoded = Video::empty();
            while decoder.receive_frame(&mut decoded).is_ok() {
                let mut frame = Video::empty();
                if let Some(pts) = decoded.timestamp() {
                    if pts < next_pts {
                        continue;
                    }
                    if pts > end_pts {
                        break;
                    }
                    next_pts += interval;
                } else {
                    return Err(anyhow::anyhow!("No timestamp"));
                }
                scaler.run(&decoded, &mut frame)?;
                let image_buffer = ImageBuffer::<image::Rgb<u8>, _>::from_raw(
                    frame.width(),
                    frame.height(),
                    frame.data(0).to_vec(),
                )
                .ok_or("Failed to create image buffer")
                .unwrap();

                let mut jpeg_file =
                    fs::File::create(format!("output/capture/frame_{:04}.jpg", frame_count))?;
                // println!("Writing frame to file: frame_{:04}.jpg", frame_count);
                let mut jpeg_data = Vec::new();
                let mut encoder = jpeg::JpegEncoder::new_with_quality(&mut jpeg_data, 100);
                encoder.encode(
                    &image_buffer,
                    image_buffer.width(),
                    image_buffer.height(),
                    image::ExtendedColorType::Rgb8,
                )?;
                jpeg_file.write_all(jpeg_data.as_slice())?;

                let base64_frame = BASE64_STANDARD.encode(jpeg_data);
                base64_frames.push("data:image/jpeg;base64,".to_owned() + &base64_frame);
                frame_count += 1;
            }
            Ok(())
        };

    for (stream, packet) in input.packets() {
        if stream.index() == video_stream_index {
            decoder.send_packet(&packet)?;
            receive_and_process_decoded_frames(&mut decoder)?;
        }
    }
    decoder.send_eof()?;
    receive_and_process_decoded_frames(&mut decoder)?;

    Ok(base64_frames)
}

enum FrameWrapper<'a> {
    Video(&'a Video),
    Audio(&'a Audio),
}

impl FrameWrapper<'_> {
    fn as_video(&self) -> anyhow::Result<&Video> {
        match self {
            &FrameWrapper::Video(frame) => Ok(frame),
            _ => anyhow::Result::Err(anyhow::anyhow!("Frame is not a video frame")),
        }
    }

    fn as_audio(&self) -> anyhow::Result<&Audio> {
        match self {
            &FrameWrapper::Audio(frame) => Ok(frame),
            _ => anyhow::Result::Err(anyhow::anyhow!("Frame is not an audio frame")),
        }
    }
}

trait Transcoder {
    fn flush_filter_graph(&mut self) -> anyhow::Result<()> {
        Ok(())
    }

    fn receive_and_process_filtered_frames(
        &mut self,
        _output: &mut format::context::Output,
        _output_stream_time_base: Rational,
    ) -> anyhow::Result<()> {
        Ok(())
    }

    fn send_packet_to_decoder(&mut self, packet: &Packet) -> anyhow::Result<()>;

    fn send_eof_to_decoder(&mut self) -> anyhow::Result<()>;

    fn receive_and_process_decoded_frames(
        &mut self,
        output: &mut format::context::Output,
        output_stream_time_base: Rational,
    ) -> anyhow::Result<()>;

    fn send_frame_to_encoder(&mut self, frame_wrapper: FrameWrapper) -> anyhow::Result<()>;

    fn send_eof_to_encoder(&mut self) -> anyhow::Result<()>;

    fn receive_and_process_encoded_packets(
        &mut self,
        output: &mut format::context::Output,
        output_stream_time_base: Rational,
    ) -> anyhow::Result<()>;
}

struct VideoTranscoder {
    output_stream_index: usize,
    decoder: decoder::Video,
    encoder: encoder::Video,
    input_time_base: Rational,
    start_sec: i64,
}

impl VideoTranscoder {
    fn new(
        input_stream: &format::stream::Stream,
        output: &mut format::context::Output,
        output_stream_index: usize,
        start_sec: i64,
    ) -> anyhow::Result<Self> {
        let global_header = output
            .format()
            .flags()
            .contains(format::Flags::GLOBAL_HEADER);
        let codec_params = input_stream.parameters();
        let decoder = codec::context::Context::from_parameters(codec_params)?
            .decoder()
            .video()?;

        let codec = encoder::find(codec::Id::H264);
        let mut output_stream = output.add_stream(codec)?;
        let mut encoder = codec::context::Context::new_with_codec(
            codec.ok_or(anyhow::anyhow!(Error::InvalidData))?,
        )
        .encoder()
        .video()?;
        encoder.set_height(decoder.height());
        encoder.set_width(decoder.width());
        encoder.set_aspect_ratio(decoder.aspect_ratio());
        encoder.set_format(decoder.format());
        encoder.set_frame_rate(decoder.frame_rate());
        encoder.set_time_base(input_stream.time_base());
        output_stream.set_parameters(&encoder);

        if global_header {
            encoder.set_flags(codec::Flags::GLOBAL_HEADER);
        }

        let mut opts = Dictionary::new();
        opts.set("preset", "medium");

        let opened_encoder = encoder.open_with(opts)?;
        output_stream.set_parameters(&opened_encoder);

        Ok(Self {
            output_stream_index,
            decoder,
            encoder: opened_encoder,
            input_time_base: input_stream.time_base(),
            start_sec,
        })
    }
}

impl Transcoder for VideoTranscoder {
    fn send_packet_to_decoder(&mut self, packet: &Packet) -> anyhow::Result<()> {
        self.decoder
            .send_packet(packet)
            .map_err(anyhow::Error::from)
    }

    fn send_eof_to_decoder(&mut self) -> anyhow::Result<()> {
        self.decoder.send_eof().map_err(anyhow::Error::from)
    }

    fn receive_and_process_decoded_frames(
        &mut self,
        output: &mut format::context::Output,
        output_stream_time_base: Rational,
    ) -> anyhow::Result<()> {
        let mut frame = Video::empty();
        let start_pts = self.start_sec.rescale((1, 1), self.input_time_base);
        while self.decoder.receive_frame(&mut frame).is_ok() {
            let timestamp = frame.timestamp().ok_or(anyhow::anyhow!("No timestamp"))?;
            frame.set_pts(Some(timestamp - start_pts));
            frame.set_kind(picture::Type::None);
            self.send_frame_to_encoder(FrameWrapper::Video(&frame))?;
            self.receive_and_process_encoded_packets(output, output_stream_time_base)?;
        }
        Ok(())
    }

    fn send_frame_to_encoder(&mut self, frame_wrapper: FrameWrapper) -> anyhow::Result<()> {
        self.encoder
            .send_frame(frame_wrapper.as_video()?)
            .map_err(anyhow::Error::from)
    }

    fn send_eof_to_encoder(&mut self) -> anyhow::Result<()> {
        self.encoder.send_eof().map_err(anyhow::Error::from)
    }

    fn receive_and_process_encoded_packets(
        &mut self,
        output: &mut format::context::Output,
        output_stream_time_base: Rational,
    ) -> anyhow::Result<()> {
        let mut packet = Packet::empty();
        while self.encoder.receive_packet(&mut packet).is_ok() {
            packet.set_stream(self.output_stream_index);
            packet.rescale_ts(self.input_time_base, output_stream_time_base);
            packet.write_interleaved(output)?;
        }
        Ok(())
    }
}

struct AudioTranscoder {
    output_stream_index: usize,
    decoder: decoder::Audio,
    encoder: encoder::Audio,
    filter_graph: filter::Graph,
    input_time_base: Rational,
    start_sec: i64,
}

impl AudioTranscoder {
    fn new(
        input_stream: &format::stream::Stream,
        output: &mut format::context::Output,
        output_stream_index: usize,
        filter_spec: &str,
        start_sec: i64,
    ) -> anyhow::Result<Self> {
        let global_header = output
            .format()
            .flags()
            .contains(format::Flags::GLOBAL_HEADER);
        let codec_params = input_stream.parameters();
        let mut decoder = codec::context::Context::from_parameters(codec_params)?
            .decoder()
            .audio()?;

        if global_header {
            decoder.set_flags(codec::Flags::GLOBAL_HEADER);
        }

        let codec = encoder::find(codec::Id::AAC)
            .ok_or(anyhow::anyhow!(Error::EncoderNotFound))?
            .audio()?;
        let mut output_stream = output.add_stream(codec)?;
        let context = codec::context::Context::from_parameters(output_stream.parameters())?;
        let mut encoder = context.encoder().audio()?;

        if global_header {
            encoder.set_flags(codec::Flags::GLOBAL_HEADER);
        }

        let channel_layout = codec
            .channel_layouts()
            .map(|layouts| layouts.best(decoder.channel_layout().channels()))
            .unwrap_or(channel_layout::ChannelLayout::STEREO);

        encoder.set_channel_layout(channel_layout);
        encoder.set_rate(decoder.rate() as _);
        encoder.set_format(
            codec
                .formats()
                .ok_or(anyhow::anyhow!("Unknown supported formats"))?
                .next()
                .ok_or(anyhow::anyhow!("Failed to get sample format"))?,
        );
        encoder.set_bit_rate(decoder.bit_rate());
        encoder.set_max_bit_rate(decoder.max_bit_rate());
        encoder.set_time_base(decoder.time_base());
        output_stream.set_time_base(decoder.time_base());

        let opened_encoder = encoder.open_as(codec)?;
        output_stream.set_parameters(&opened_encoder);

        let filter_graph = Self::filter_graph(filter_spec, &decoder, &opened_encoder)?;

        Ok(Self {
            output_stream_index,
            decoder,
            encoder: opened_encoder,
            filter_graph,
            input_time_base: input_stream.time_base(),
            start_sec,
        })
    }

    fn filter_graph(
        spec: &str,
        decoder: &codec::decoder::Audio,
        encoder: &codec::encoder::Audio,
    ) -> anyhow::Result<filter::Graph> {
        let mut filter_graph = filter::Graph::new();

        let args = format!(
            "time_base={}:sample_rate={}:sample_fmt={}:channel_layout=0x{:x}",
            decoder.time_base(),
            decoder.rate(),
            decoder.format().name(),
            decoder.channel_layout().bits()
        );

        filter_graph.add(
            &filter::find("abuffer").ok_or(anyhow::anyhow!("Failed to find filter"))?,
            "in",
            &args,
        )?;
        filter_graph.add(
            &filter::find("abuffersink").ok_or(anyhow::anyhow!("Failed to find filter"))?,
            "out",
            "",
        )?;

        {
            let mut out = filter_graph
                .get("out")
                .ok_or(anyhow::anyhow!("Failed to get filter"))?;
            out.set_sample_format(encoder.format());
            out.set_channel_layout(encoder.channel_layout());
            out.set_sample_rate(encoder.rate());
        }

        filter_graph.output("in", 0)?.input("out", 0)?.parse(spec)?;
        filter_graph.validate()?;

        println!("Filter graph: {}", filter_graph.dump());

        if let Some(codec) = encoder.codec() {
            if !codec
                .capabilities()
                .contains(ffmpeg::codec::capabilities::Capabilities::VARIABLE_FRAME_SIZE)
            {
                filter_graph
                    .get("out")
                    .ok_or(anyhow::anyhow!("Failed to get filter"))?
                    .sink()
                    .set_frame_size(encoder.frame_size());
            }
        }

        Ok(filter_graph)
    }

    fn add_frame_to_filter_graph(&mut self, frame: &Frame) -> anyhow::Result<()> {
        self.filter_graph
            .get("in")
            .ok_or(anyhow::anyhow!("Failed to get filter"))?
            .source()
            .add(frame)
            .map_err(|e| anyhow::anyhow!(e))
    }
}

impl Transcoder for AudioTranscoder {
    fn flush_filter_graph(&mut self) -> anyhow::Result<()> {
        self.filter_graph
            .get("in")
            .ok_or(anyhow::anyhow!("Failed to get filter"))?
            .source()
            .flush()
            .map_err(|e| anyhow::anyhow!(e))
    }

    fn receive_and_process_filtered_frames(
        &mut self,
        output: &mut format::context::Output,
        output_stream_time_base: Rational,
    ) -> anyhow::Result<()> {
        let mut frame = Audio::empty();
        while self
            .filter_graph
            .get("out")
            .ok_or(anyhow::anyhow!("Failed to get filter"))?
            .sink()
            .frame(&mut frame)
            .is_ok()
        {
            self.send_frame_to_encoder(FrameWrapper::Audio(&frame))?;
            self.receive_and_process_encoded_packets(output, output_stream_time_base)?;
        }
        Ok(())
    }

    fn send_packet_to_decoder(&mut self, packet: &Packet) -> anyhow::Result<()> {
        self.decoder
            .send_packet(packet)
            .map_err(anyhow::Error::from)
    }

    fn send_eof_to_decoder(&mut self) -> anyhow::Result<()> {
        self.decoder.send_eof().map_err(anyhow::Error::from)
    }

    fn receive_and_process_decoded_frames(
        &mut self,
        output: &mut format::context::Output,
        output_stream_time_base: Rational,
    ) -> anyhow::Result<()> {
        let mut frame = Audio::empty();
        let start_pts = self.start_sec.rescale((1, 1), self.input_time_base);
        while self.decoder.receive_frame(&mut frame).is_ok() {
            let timestamp = frame.timestamp().ok_or(anyhow::anyhow!("No timestamp"))?;
            frame.set_pts(Some(timestamp - start_pts));
            self.add_frame_to_filter_graph(&frame)?;
            self.receive_and_process_filtered_frames(output, output_stream_time_base)?;
        }
        Ok(())
    }

    fn send_frame_to_encoder(&mut self, frame_wrapper: FrameWrapper) -> anyhow::Result<()> {
        self.encoder
            .send_frame(frame_wrapper.as_audio()?)
            .map_err(anyhow::Error::from)
    }

    fn send_eof_to_encoder(&mut self) -> anyhow::Result<()> {
        self.encoder.send_eof().map_err(anyhow::Error::from)
    }

    fn receive_and_process_encoded_packets(
        &mut self,
        output: &mut format::context::Output,
        output_stream_time_base: Rational,
    ) -> anyhow::Result<()> {
        let mut packet = Packet::empty();
        while self.encoder.receive_packet(&mut packet).is_ok() {
            packet.set_stream(self.output_stream_index);
            packet.rescale_ts(self.input_time_base, output_stream_time_base);
            packet.write_interleaved(output)?;
        }
        Ok(())
    }
}

pub(crate) fn transcode(
    input_path: &Path,
    overlay_audio_path: &Path,
    output_path: &Path,
    start_sec: i64,
    duration_sec: i64,
) -> anyhow::Result<()> {
    let mut input = format::input(input_path)?;
    let mut output = format::output(&output_path)?;
    let mut transcoders: HashMap<i32, Box<dyn Transcoder>> = HashMap::new();

    let overlay_audio_filter_spec = if fs::exists(overlay_audio_path)? {
        format!(
            "amovie={},atempo=1.25,volume=1.2 [ov]; [in]volume=0.8 [in_vol]; [in_vol][ov] amix=inputs=2:duration=shortest [out]",
            overlay_audio_path
                .to_str()
                .ok_or(anyhow::anyhow!("Invalid comment audio path"))?
        )
    } else {
        "anull".to_owned()
    };

    println!("Overlay audio filter spec: {}", overlay_audio_filter_spec);

    format::context::input::dump(
        &input,
        0,
        Some(input_path.to_str().ok_or(anyhow::anyhow!("Invalid path"))?),
    );

    let start_pos = start_sec.rescale((1, 1), rescale::TIME_BASE);
    input.seek(start_pos, ..start_pos)?;

    let mut stream_mapping = vec![0_i32; input.nb_streams() as _];
    let mut input_stream_time_base = vec![Rational(0, 0); input.nb_streams() as _];
    let mut output_stream_time_base = vec![Rational(0, 0); input.nb_streams() as _];
    let mut output_stream_index = 0;
    for (ist_index, ist) in input.streams().enumerate() {
        let ist_medium = ist.parameters().medium();
        if ist_medium != media::Type::Audio
            && ist_medium != media::Type::Video
            && ist_medium != media::Type::Subtitle
        {
            stream_mapping[ist_index] = -1;
            continue;
        }
        stream_mapping[ist_index] = output_stream_index;
        input_stream_time_base[ist_index] = ist.time_base();
        if ist_medium == media::Type::Video {
            let transcoder = Box::new(VideoTranscoder::new(
                &ist,
                &mut output,
                output_stream_index as _,
                start_sec,
            )?);
            transcoders.insert(ist_index as i32, transcoder);
        } else if ist_medium == media::Type::Audio {
            let transcoder = Box::new(AudioTranscoder::new(
                &ist,
                &mut output,
                output_stream_index as _,
                overlay_audio_filter_spec.as_str(),
                start_sec,
            )?);
            transcoders.insert(ist_index as i32, transcoder);
        }
        output_stream_index += 1;
    }

    output.set_metadata(input.metadata().to_owned());
    format::context::output::dump(
        &output,
        0,
        Some(
            output_path
                .to_str()
                .ok_or(anyhow::anyhow!("Invalid path"))?,
        ),
    );
    output.write_header()?;

    for (ost_index, _) in output.streams().enumerate() {
        output_stream_time_base[ost_index] = output
            .stream(ost_index)
            .ok_or(anyhow::anyhow!(Error::StreamNotFound))?
            .time_base();
    }

    for (ist, mut packet) in input.packets() {
        let ist_index = ist.index();
        let ost_index = stream_mapping[ist_index];
        if ost_index < 0 {
            continue;
        }
        let end_pts = (start_sec + duration_sec).rescale((1, 1), ist.time_base());
        let pts = packet.pts().ok_or(anyhow::anyhow!("No pts"))?;
        if pts >= end_pts {
            break;
        }

        let ost_time_base = output_stream_time_base[ost_index as usize];
        match transcoders.get_mut(&(ist_index as i32)) {
            Some(transcoder) => {
                transcoder.send_packet_to_decoder(&packet)?;
                transcoder.receive_and_process_decoded_frames(&mut output, ost_time_base)?;
            }
            None => {
                packet.rescale_ts(input_stream_time_base[ist_index], ost_time_base);
                packet.write_interleaved(&mut output)?;
            }
        }
    }

    for (ost_index, transcoder) in transcoders.iter_mut() {
        let ost_time_base = output_stream_time_base[*ost_index as usize];
        transcoder.send_eof_to_decoder()?;
        transcoder.receive_and_process_decoded_frames(&mut output, ost_time_base)?;
        transcoder.flush_filter_graph()?;
        transcoder.receive_and_process_filtered_frames(&mut output, ost_time_base)?;
        transcoder.send_eof_to_encoder()?;
        transcoder.receive_and_process_encoded_packets(&mut output, ost_time_base)?;
    }

    output.write_trailer()?;

    Ok(())
}
