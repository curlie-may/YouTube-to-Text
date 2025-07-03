# YouTube-to-Text
An AI Agent Program to convert YouTube videos to text

As the title says, this program converts YouTube video to text.  Yes, I know, there are several applications out there that already do this such as Google AI Studio. But those tend to be cluttered and sometimes difficult to navigate.  I wanted this to have a clean simple user interface. And I also wanted the experience of developing an app using my newly-learned AI Agent skills.  I used Claude to vibe-code the application.  After the captions are extracted, an AI Agent proofreads the transcript to fix grammar and modifies it so that the output is technically correct, but doesn't change the natural flow of the speaker.

The text extraction relies on the video's closed captions (CC).  This is unlike Google Studi AI that downloads the video and extracts text.  The Google AI Studio approach is much more accurate, BUT, for non-Google affiliates this can lead to a liability issue. YouTube videos are owned by Google.  Therefore, Google Studio AI has license to process YouTube videos however needed.  Everyone else is restricted.  It's not to say you can't download these videos, it just would be against the usage rules.

With that being said, what have I learned from building this app?  The answer is A LOT.

I learned that building this transcription tool is MUCH more difficult than I originally would have thought.  The captions are not grammatically clean, for example, there is no punctuation and no capitalizations to indicate the beginning of a sentence.  This task was assigned to an AI Agent. I learned that the Agent prompt is critical.  The original transcript output had run-on and incomplete sentences. Simply adding to the prompt "A sentece should have at minimum a subject and a verb" dramatically improved the sentence structure.

The output needed to be chunked or else I ran into OpenAI token limits. Information chunking seems to be a standard approach for AI applications, however, I learned that it's difficult to implement.  The size of the chunks need to be optimized. If the chunks are too big you run into token limits for a single AI call.  If they're too small it takes forever to run the program.  Also, at the interface of the chunks the sentence need to be stitched together.  This created issues with sentence overlap and redundancy.  The Agent was critical in interpreting the sentence structure so it made sense at these junctures.

For GUI Feedback link located on the GUI, I learned how to implement an SMTP call so I would receive an email when a user filled out the form. This involved creating a password for the app. 

I learned how to deploy the app using Render.  

I plan to use the app to transcribe educational videos so I can read them at night before bed.  I hope others find the app useful and that it makes the world a tiny bit better.         
