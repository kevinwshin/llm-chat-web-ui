from typing import List
import modal

from modal_base import stub, Message

MODEL_DIR = "/root/model"


def download_model_to_folder():
    from huggingface_hub import snapshot_download
    import os

    os.makedirs(MODEL_DIR, exist_ok=True)

    snapshot_download(
        "KoboldAI/LLaMA2-13B-Psyfighter2",
        local_dir=MODEL_DIR,
        token=os.environ["HUGGING_FACE_HUB_TOKEN"],
    )


image = (
    modal.Image.from_registry("nvcr.io/nvidia/pytorch:22.12-py3")
    .pip_install("torch==2.0.1", index_url="https://download.pytorch.org/whl/cu118")
    .pip_install(
        "https://github.com/vllm-project/vllm/releases/download/v0.2.7/vllm-0.2.7+cu118-cp38-cp38-manylinux1_x86_64.whl",
        "typing-extensions==4.5.0",  # >=4.6 causes typing issues
    )
    # Use the barebones hf-transfer package for maximum download speeds. No progress bar, but expect 700MB/s.
    .pip_install("hf-transfer~=0.1")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .run_function(
        download_model_to_folder,
        secret=modal.Secret.from_name("llm-chat-secret"),
        timeout=60 * 20,
    )
)


@stub.cls(
    gpu=modal.gpu.A100(),
    container_idle_timeout=60,
    image=image,
    secret=modal.Secret.from_name("llm-chat-secret"),
)
class VLLMHFModel:
    def __init__(self, temperature: float, system_prompt: str):
        self.temperature = temperature
        self.system_prompt = system_prompt

    def __enter__(self):
        from vllm import LLM

        # Load the model. Tip: MPT models may require `trust_remote_code=true`.
        self.llm = LLM(MODEL_DIR)
        self.template = """<s>[INST] <<SYS>>
{system}
<</SYS>>

{user} [/INST] """

    @modal.method()
    def generate(self, chat: List[Message]):
        from vllm import SamplingParams
        import json

        question = chat[-1].content
        prompts = [self.template.format(system=self.system_prompt, user=question)]
        print(prompts)
        sampling_params = SamplingParams(
            temperature=max(self.temperature, 0.01),
            top_p=1,
            max_tokens=1000,
            presence_penalty=1.15,
        )
        result = self.llm.generate(prompts, sampling_params)
        for output in result:
            yield json.dumps({"content": output.outputs[0].text}) + "\n"
