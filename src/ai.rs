use std::path::Path;

use async_openai::error::OpenAIError;
use async_openai::types::{
    ChatCompletionRequestMessage, ChatCompletionRequestMessageContentPartImageArgs,
    ChatCompletionRequestMessageContentPartTextArgs, ChatCompletionRequestUserMessageArgs,
    ChatCompletionRequestUserMessageContent, ChatCompletionRequestUserMessageContentPart,
    CreateChatCompletionRequestArgs, CreateSpeechRequestArgs, ImageUrlArgs, SpeechModel, Voice,
};
use async_openai::Client;

pub(crate) async fn annotation_frames(prompt: &str, frames: Vec<String>) -> anyhow::Result<String> {
    let request = CreateChatCompletionRequestArgs::default()
        .model("gpt-4o")
        .max_tokens(512_u32)
        .messages([ChatCompletionRequestMessage::User(
            ChatCompletionRequestUserMessageArgs::default()
                .content(ChatCompletionRequestUserMessageContent::Array(
                    [
                        vec![ChatCompletionRequestUserMessageContentPart::Text(
                            ChatCompletionRequestMessageContentPartTextArgs::default()
                                .text(prompt)
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
    let response = tokio::time::timeout(
        tokio::time::Duration::from_secs(300),
        ai_client.chat().create(request),
    )
    .await??;
    response.choices[0]
        .clone()
        .message
        .content
        .ok_or(anyhow::anyhow!("No content in response from OpenAI"))
}

pub(crate) async fn audio_speech(text: &str, output_path: &Path) -> anyhow::Result<()> {
    let request = CreateSpeechRequestArgs::default()
        .input(text)
        .voice(Voice::Nova)
        .model(SpeechModel::Tts1Hd)
        .build()?;

    let client = Client::new();
    let response = tokio::time::timeout(
        tokio::time::Duration::from_secs(120),
        client.audio().speech(request),
    )
    .await??;
    response.save(output_path).await?;
    Ok(())
}
