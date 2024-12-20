mod ai;
mod video;

use clap::Parser;
use std::fs;
use std::path::{Path, PathBuf};
#[derive(Parser)]
#[command(name = "annotai")]
#[command(about = "Annotate videos using OpenAI's GPT-4o", long_about = None)]
struct Cli {
    input_file: PathBuf,
    #[arg(short, long)]
    prompt: String,
    #[arg(short, long, default_value_t = 0)]
    start_sec: i64,
    #[arg(short, long, default_value_t = 30)]
    duration_sec: i64,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    fs::exists(&cli.input_file)?;
    fs::create_dir_all("output")?;

    let capture_interval_msec = 500;
    video::init();
    let frames = video::capture_base64(
        cli.input_file.as_path(),
        cli.start_sec,
        cli.duration_sec,
        capture_interval_msec,
    )?;

    println!("Captured frames: {}", frames.len());

    let comment = ai::annotation_frames(&cli.prompt, frames).await?;

    println!("AI Comment: {}", comment);

    let comment_audio_path = Path::new("output/comment.mp3");
    ai::audio_speech(&comment, comment_audio_path).await?;

    let transcoded_path = Path::new("output/transcoded.mp4");
    video::transcode(
        cli.input_file.as_path(),
        comment_audio_path,
        transcoded_path,
        cli.start_sec,
        cli.duration_sec * 2,
    )?;

    Ok(())
}
