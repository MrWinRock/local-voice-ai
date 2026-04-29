import logging
import os
from typing import Any

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    cli,
    function_tool,
    RunContext,
)
from livekit.plugins import silero, openai
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")

load_dotenv(".env.local")

class Assistant(Agent):
    def __init__(self) -> None:
        language = os.getenv("LANGUAGE", "th")
        if language == "th":
            instructions = """คุณเป็นผู้ช่วย AI ด้านเสียงที่มีประโยชน์ ผู้ใช้กำลังโต้ตอบกับคุณผ่านเสียง แม้ว่าคุณจะรับรู้การสนทนาเป็นข้อความ
            คุณช่วยเหลือผู้ใช้อย่างกระตือรือร้นด้วยการให้ข้อมูลจากความรู้ที่ครอบคลุมของคุณ
            ตอบสนองด้วยภาษาไทยเสมอ คำตอบของคุณต้องกระชับ ตรงประเด็น และไม่มีการจัดรูปแบบที่ซับซ้อนหรือเครื่องหมายวรรคตอนที่ซับซ้อน รวมถึงอีโมจิ เครื่องหมายดอกจัน หรือสัญลักษณ์อื่นๆ
            คุณอยากรู้อยากเห็น เป็นมิตร และมีอารมณ์ขัน"""
        else:
            instructions = """You are a helpful voice AI assistant. The user is interacting with you via voice, even if you perceive the conversation as text.
            You eagerly assist users with their questions by providing information from your extensive knowledge.
            Your responses are concise, to the point, and without any complex formatting or punctuation including emojis, asterisks, or other symbols.
            You are curious, friendly, and have a sense of humor."""
        super().__init__(instructions=instructions)

    @function_tool()
    async def multiply_numbers(
        self,
        context: RunContext,
        number1: int,
        number2: int,
    ) -> dict[str, Any]:
        """Multiply two numbers.
        
        Args:
            number1: The first number to multiply.
            number2: The second number to multiply.
        """

        return f"The product of {number1} and {number2} is {number1 * number2}."

server = AgentServer()

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()

server.setup_fnc = prewarm

@server.rtc_session()
async def my_agent(ctx: JobContext):
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    llama_model = os.getenv("LLAMA_MODEL", "qwen3-4b")
    llama_base_url = os.getenv("LLAMA_BASE_URL", "http://llama_cpp:11434/v1")

    language = os.getenv("LANGUAGE", "th")

    stt_provider = os.getenv("STT_PROVIDER", "whisper").lower()
    if stt_provider == "nemotron":
        default_stt_base_url = "http://nemotron:8000/v1"
        default_stt_model = "nemotron-speech-streaming"
    else:
        default_stt_base_url = "http://whisper:80/v1"
        default_stt_model = "Systran/faster-whisper-small"

    stt_base_url = os.getenv("STT_BASE_URL", default_stt_base_url)
    stt_model = os.getenv("STT_MODEL", default_stt_model)
    stt_api_key = os.getenv("STT_API_KEY", "no-key-needed")

    default_tts_voice = "th-TH-PremwadeeNeural" if language == "th" else "af_nova"
    tts_voice = os.getenv("TTS_VOICE", default_tts_voice)

    logger.info(
        "Starting agent with STT provider=%s model=%s base_url=%s language=%s TTS voice=%s",
        stt_provider,
        stt_model,
        stt_base_url,
        language,
        tts_voice,
    )

    if language == "th":
        tts_plugin = openai.TTS(
            base_url="http://thai-tts:8881/v1",
            # base_url="http://localhost:8881/v1", # uncomment for local testing
            model="edge-tts",
            voice=tts_voice,
            api_key="no-key-needed",
        )
    else:
        tts_plugin = openai.TTS(
            base_url="http://kokoro:8880/v1",
            # base_url="http://localhost:8880/v1", # uncomment for local testing
            model="kokoro",
            voice=tts_voice,
            api_key="no-key-needed",
        )

    session = AgentSession(
        stt=openai.STT(
            base_url=stt_base_url,
            # base_url="http://localhost:11437/v1", # uncomment for local testing
            model=stt_model,
            api_key=stt_api_key,
            language=language,
        ),
        llm=openai.LLM(
            base_url=llama_base_url,
            # base_url="http://localhost:11436/v1", # uncomment for local testing
            model=llama_model,
            api_key="no-key-needed"
        ),
        tts=tts_plugin,
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )

    await session.start(
        agent=Assistant(),
        room=ctx.room,
    )

    await ctx.connect()
    
    await session.generate_reply(
        instructions="สวัสดี คุณต้องการให้ช่วยอะไรไหม"
    )

if __name__ == "__main__":
    cli.run_app(server)
