mod capture;

use async_openai::error::OpenAIError;
use async_openai::types::{
    ChatCompletionRequestMessage, ChatCompletionRequestMessageContentPartImageArgs,
    ChatCompletionRequestMessageContentPartTextArgs, ChatCompletionRequestUserMessageArgs,
    ChatCompletionRequestUserMessageContent, ChatCompletionRequestUserMessageContentPart,
    CreateChatCompletionRequestArgs, ImageUrlArgs,
};
use async_openai::Client;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let ai_client = Client::new();
    capture::init();
    let frames = capture::capture_base64("input.mp4", 10000, 2000, 1000)?;

    let request = CreateChatCompletionRequestArgs::default()
        .model("gpt-4o")
        .max_tokens(512_u32)
        .messages([
            ChatCompletionRequestMessage::User(
                ChatCompletionRequestUserMessageArgs::default().content(
                    ChatCompletionRequestUserMessageContent::Array(
                        [
                            vec![ChatCompletionRequestUserMessageContentPart::Text(
                                ChatCompletionRequestMessageContentPartTextArgs::default()
                                .text("These are frames of a video. Create a short voiceover script in the style of David Attenborough. Only include the narration.")
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
                    )
                )
                .build()?
            )
        ])
        .build()?;

    let response = ai_client.chat().create(request).await?;
    println!("{:?}", response);

    Ok(())
}
