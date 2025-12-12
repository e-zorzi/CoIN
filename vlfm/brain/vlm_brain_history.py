from dataclasses import dataclass
from typing import Literal

from colorama import Fore
from colorama import init as init_colorama
from PIL import Image

from vlfm.brain.LLM import VllmLLM
from vlfm.utils.prompts import (
    SINGLEMODEL_PROMPT_CHOICES,
    SINGLEMODEL_PROMPT_ONLY_CHOICES,
    SINGLEMODEL_PROMPT_ONLY_SCORE,
    SINGLEMODEL_PROMPT_SCORE,
)

init_colorama(autoreset=True)
import numpy as np
from vlfm.utils.prompts import LLava_REDUCE_FALSE_POSITIVE


class Conversation:
    def __init__(self) -> None:
        self.conversation = []

    def add_message(self, role, text, image):
        content = {"type": "image", "image": image, "text": text}
        self.conversation.append({"role": role, "content": [content]})

    def get_convrersation(self):
        return self.conversation

    def reset(self):
        self.conversation = []


class VLM_History:
    """
    This class store the connector for the vlm (LLaVa in this case) and it's past conversation.
    """

    def __init__(self, llava_client) -> None:
        self.llava_client = llava_client
        self.ep_id = None
        self.conversation = Conversation()

    def get_description_of_the_image(self, image, prompt):
        output, _ = self.llava_client.ask(image, prompt=prompt)
        print(Fore.LIGHTMAGENTA_EX + "[INFO: On-board VLM] " + output)
        return output

    def reduce_detector_false_positive(
        self, detected_image: np.ndarray, target_object: str, get_logits: bool = False
    ) -> str:
        prompt = LLava_REDUCE_FALSE_POSITIVE.format(target_object=target_object)
        response = self.llava_client.ask(detected_image, prompt=prompt, return_token_likelihood=get_logits)
        return response

    def reset(self):
        self.ep_id = None
        self.conversation.reset()


class VLM_client:
    def __init__(self, model_id: str, annotation_type: Literal["score", "choice"], only_annotation: bool = False):
        self._client = VllmLLM(model_id=model_id)
        self.annotation_type = annotation_type
        self.only_annotation = only_annotation

        if annotation_type == "score":
            self.prompt = SINGLEMODEL_PROMPT_SCORE if not only_annotation else SINGLEMODEL_PROMPT_ONLY_SCORE
            # A bit arbitrary as the score is not 'ufficialy' linked to any scale (just how likely it is that the image
            # matches the description)
            self.success_scores = [4, 5]
            self.indecisive_scores = [3]
            self.failure_scores = [1, 2]
        elif annotation_type == "choice":
            self.prompt = SINGLEMODEL_PROMPT_CHOICES if not only_annotation else SINGLEMODEL_PROMPT_ONLY_CHOICES
            # Not arbitrary, the model was trained with this scale
            self.success_scores = [2]
            self.indecisive_scores = [1]
            self.failure_scores = [0]
        else:
            raise NotImplementedError

    def ask(self, image, task):
        # As a sanity check, check whether the passed task is always the same
        # for each episode (in reset() it is set to None)
        if self._initial_task is None:
            self._initial_task = task
        else:
            assert task == self._initial_task
        # Parse the returned reasoning
        try:
            # Should have form <motivation>Motivation</motivation><score>Score or choice</score>
            reasoning = self._client.image_text_chat(
                self.prompt.format(USER_TASK=task),
                image,
            )

            splitted_reasoning = reasoning.split("<score>")
            score = int(splitted_reasoning[1].rstrip("</score>"))
            reasoning = splitted_reasoning[0].lstrip("<motivation>").rstrip("</motivation>")
        except Exception as e:  # noqa
            # The format was wrong
            print("[ERROR] " + str(e))
            reasoning = "<|BAD REASONING|>"
            score = self.indecisive_scores[0]
        print(Fore.LIGHTMAGENTA_EX + "[INFO: On-board VLM] " + reasoning + "\nScore: [" + str(score) + "]" + Fore.WHITE)

        return (reasoning, score)

    def reset(self):
        self._initial_task = None
