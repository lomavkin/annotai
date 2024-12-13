mod capture;

use std::fs;
use std::path::PathBuf;

use async_openai::error::OpenAIError;
use async_openai::types::{
    ChatCompletionRequestMessage, ChatCompletionRequestMessageContentPartImageArgs,
    ChatCompletionRequestMessageContentPartTextArgs, ChatCompletionRequestUserMessageArgs,
    ChatCompletionRequestUserMessageContent, ChatCompletionRequestUserMessageContentPart,
    CreateChatCompletionRequestArgs, ImageUrlArgs,
};
use async_openai::Client;
use clap::Parser;

#[derive(Parser)]
#[command(name = "annotai")]
#[command(about = "Annotate videos using OpenAI's GPT-4o", long_about = None)]
struct Cli {
    input_file: PathBuf,
    #[arg(short, long)]
    prompt: String,
    #[arg(short, long, default_value_t = 0)]
    start_time_ms: u32,
    #[arg(short, long, default_value_t = 1000)]
    duration_ms: u32,
    #[arg(short, long, default_value_t = 1000)]
    capture_interval_ms: u32,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    fs::exists(&cli.input_file)?;

    capture::init();
    let frames = capture::capture_base64(
        cli.input_file
            .as_path()
            .to_str()
            .ok_or(anyhow::anyhow!("Invalid path"))?,
        cli.start_time_ms,
        cli.duration_ms,
        cli.capture_interval_ms,
    )?;

    let request = CreateChatCompletionRequestArgs::default()
        .model("gpt-4o")
        .max_tokens(512_u32)
        .messages([ChatCompletionRequestMessage::User(
            ChatCompletionRequestUserMessageArgs::default()
                .content(ChatCompletionRequestUserMessageContent::Array(
                    [
                        vec![ChatCompletionRequestUserMessageContentPart::Text(
                            ChatCompletionRequestMessageContentPartTextArgs::default()
                                .text(cli.prompt)
                                .build()?,
                        )],
                        frames
                            .into_iter()
                            .map(|frame| -> Result<_, OpenAIError> {
                                Ok(ChatCompletionRequestUserMessageContentPart::ImageUrl(
                                    ChatCompletionRequestMessageContentPartImageArgs::default()
                                        .image_url(ImageUrlArgs::default().url(frame).build()?)
                                        .build()?,
                                ))
                            })
                            .collect::<Result<_, _>>()?,
                    ]
                    .concat(),
                ))
                .build()?,
        )])
        .build()?;

    let ai_client = Client::new();
    let response = ai_client.chat().create(request).await?;
    println!("{:?}", response.choices[0].message.content);

    Ok(())
}
