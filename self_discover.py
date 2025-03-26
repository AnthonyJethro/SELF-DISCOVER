import os
from prompts import (
    select_prompt,
    reasoning_modules,
    adapt_prompt,
    implement_prompt,
)
from llm import LLM
from task_example import task1
import logging


def setup_logging():
    logger = logging.getLogger("__name__")
    logger.setLevel(logging.INFO)

    handler = logging.FileHandler("prompt_log.txt")
    handler.setLevel(logging.INFO)

    formatter = logging.Formatter("%(levelname)s - %(message)s")
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    return logger


logger = setup_logging()


class SelfDiscover:
    def __init__(self, task) -> None:
        # Check if local LLM should be used
        use_local_llm = os.getenv("USE_LOCAL_LLM", "false").lower() == "true"
        model_name = "Gemma3" if use_local_llm else "OpenAI"
        self.llm = LLM(model_name=model_name)
        self.actions = ["SELECT", "ADAPT", "IMPLEMENT"]
        self.task = task

    def __call__(self):
        for action in self.actions:
            print(action)
            if action == "SELECT":
                prompt = select_prompt.replace("{Task}", self.task)
                prompt = prompt.replace("{resonining_modules}", reasoning_modules)
                logger.info("SELECT PROMPT :" + prompt)
                self.selected_modules = self.llm(prompt)
                if not self.selected_modules:
                    logger.error("SELECTED_MODULES is None or empty.")
                    raise ValueError("Failed to generate SELECTED_MODULES.")
                print(self.selected_modules)

            elif action == "ADAPT":
                if not self.selected_modules:
                    logger.error("Cannot proceed to ADAPT because SELECTED_MODULES is None.")
                    raise ValueError("SELECTED_MODULES is required for ADAPT.")
                prompt = adapt_prompt.replace("{Task}", self.task)
                prompt = prompt.replace("{selected_modules}", self.selected_modules)
                logger.info("ADAPT PROMPT :" + prompt)
                self.adapted_modules = self.llm(prompt)
                if not self.adapted_modules:
                    logger.error("ADAPTED_MODULES is None or empty.")
                    raise ValueError("Failed to generate ADAPTED_MODULES.")

            elif action == "IMPLEMENT":
                if not self.adapted_modules:
                    logger.error("Cannot proceed to IMPLEMENT because ADAPTED_MODULES is None.")
                    raise ValueError("ADAPTED_MODULES is required for IMPLEMENT.")
                prompt = implement_prompt.replace("{Task}", self.task)
                prompt = prompt.replace("{adapted_modules}", self.adapted_modules)
                logger.info("IMPLEMENT PROMPT:" + prompt)
                self.reasoning_structure = self.llm(prompt)
                if not self.reasoning_structure:
                    logger.error("REASONING_STRUCTURE is None or empty.")
                    raise ValueError("Failed to generate REASONING_STRUCTURE.")


if __name__ == "__main__":
    result = SelfDiscover(task=task1)
    result()
    logger.info(f"SELECTED_MODULES : {result.selected_modules}")
    logger.info(f"ADAPTED_MODULES : {result.adapted_modules}")
    logger.info(f"REASONING_STRUCTURE : {result.reasoning_structure}")