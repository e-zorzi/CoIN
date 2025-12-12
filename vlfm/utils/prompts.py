LLaVa_TARGET_OBJECT_IS_DETECTED = """Describe the {target_object} in the provided image."""

LLava_REDUCE_FALSE_POSITIVE = """Is the object outlined with a red border in this image a {target_object}? You must answer only with Yes, No, or ?=I don't know."""

LLM_IS_THIS_THE_TARGET_IMAGE_HUMAN_FEEDBACK_ORACLE_V1 = """You are an intelligent embodied agent equipped with an RGB sensor, object detector, and a VQA model. Your task is to explore an indoor environment and find a specific target {target_object} based on the provided description.

Target {target_object} Description:
<start_target_object_description>
{target_image_description}
<end_target_object_description>

Detected {target_object} Attributes and its surroundings:
{list_of_self_questioner_detected_attributes}

Task: Answer the following question in YAML format, explaining the most significant difference between the target {target_object} and the detected {target_object}, as if you were a human.

YAML_START
Question: "Is this the target {target_object}?"
Answer: "No, this is not the {target_object} I'm looking for, <explain the difference using the most significant attribute>"
YAML_END

Provide your reasoning step-by-step, after the YAML_END tag."""


LLM_FACTS_UPDATER_AFTER_IS_THIS_TARGET_OBJECT_ORACLE_QUESTION_V1 = """
You are an intelligent embodied agent tasked with finding a specific target {target_object}.

You know the following facts about the target {target_object}:
<START_TARGET_PICTURE_FACTS> {facts_about_the_target_picture} <END_TARGET_PICTURE_FACTS> 

You recently detected another picture and asked to the human several question:
<START_OF_ORACLE_ANSWER>
{oracle_questions_answer}
<END_OF_ORACLE_ANSWER>

Task: Update the target facts with this new information. Be concise. Do not include information that are uncertain.

YAML_START
facts: <updated facts as a single text line>
YAML_END # must be present to get the information back
Provide your reasoning step-by-step, after the YAML_END tag."""


UNCERTAIN_ANSWER_CHOICE_PLACEHOLDER = "?=I don't know."
LLM_SELF_QUESTIONER_GIVEN_DISTRACTOR_DESCRIPTION = """
You are an intelligent embodied agent equipped with an RGB sensor, an object detector, and a Visual Question Answering (VQA) model. Your task is to explore an indoor environment to find a specific target {target_object}.
The detector has identified a {target_object}. The VQA model has provided the following description of the scene:

<START_OF_DESCRIPTION>
{distractor_object_description}
<END_OF_DESCRIPTION>

Based on your past interactions with the user, you know the following facts about the target picture: <START_TARGET_PICTURE_FACTS> {facts_about_the_target_picture} <END_TARGET_PICTURE_FACTS> 

Assume that the detected image description contains hallucinations. Your goal is to verify every attribute of the detected {target_object} description through questions. Formally:
- Detect possible hallucinations in the VQA model's description
- Get more information about the detected object.
Every question should be in this format: "<question content>? You must answer only with Yes, No, or {uncertain_answer_choice_placeholder}" This allows us to access likelihood the the answers.


Ensure your output follows the following format:
YAML_START # must be present to get the information back
attributes_of_the_image:
    <attribute name>: "<attribute value>" # summarize all the known attributes from the description, enclosed in " "

questions_for_detected_object: # question for the detected object, if any
    <Question number>:  "<question>? You must answer only with Yes, No, or ?=I don't know."
reasoning_for_detected_object:
    <Question number>: <reasoning>
YAML_END # must be present to get the information back

Provide your reasoning step-by-step, after the YAML_END tag."""


LMM_RETRIEVE_FACTS_FROM_DESCRIPTION = """
You are an intelligent embodied agent equipped with an RGB sensor, an object detector, and a Visual Question Answering (VQA) model. 
Your task is to explore an indoor environment to find a specific target {target_object}.
The detector has identified a {target_object}. The VQA model has provided the following description of the scene:

<START_OF_DESCRIPTION>
{distractor_object_description}
<END_OF_DESCRIPTION>

Based on your past interactions with the user, you know the following facts about the target picture: 
<START_TARGET_PICTURE_FACTS> {facts_about_the_target_picture} <END_TARGET_PICTURE_FACTS> 

Your task is to:
- ask more question to the VQA model on the detected {target_object} to maximize information gain.

Ensure your output follows the following format:

YAML_START # must be present to get the information back
attributes_of_the_image:
    <attribute name>: "<attribute value>" # summarize all the known attributes from the description, enclosed in " "
questions:
        <question_number>: "<question content>"
YAML_END # must be present to get the information back
      
Provide your reasoning step-by-step, after the YAML_END tag."""


LLM_REFINE_DETECTED_OBJECT_DESCRIPTION = """
You are an intelligent embodied agent equipped with an RGB sensor, an object detector, and a Visual Question Answering (VQA) model. 
Your task is to refine an image description based on certainty estimates and user interactions.

Scenario:
The detector has identified a scene with a {target_object}. The VQA model provided this initial scene description:

<START_OF_DESCRIPTION>
{distractor_object_description}
<END_OF_DESCRIPTION>


Questions asked and responses:
<START_QUESTION_AND_RESPONSES>
{list_questions_answers_uncertainty_labels}
<END_QUESTION_AND_RESPONSES>

Task:
Using the questions/answer pairs with uncertainty labels, refine the image description. 
Since we have to find a {target_object}, put enphasis on it. Do not include in the description information that is labeled as uncertain.

Ensure your response follows the format below:
YAML_START # must be present to get the information back
attributes_of_the_image:
    <attribute name>: "<attribute value>" # summarize all the known attributes from the description, enclosed in " "
image_description_refined: <insert refined description>  # Ensure that the string does not contain a newline (\n) after the tag image_description_refined:
YAML_END # must be present to get the information back
      
Provide your reasoning step-by-step, after the YAML_END tag."""

LLM_SIMILARITY_SCORE_AND_QUESTION_TO_TARGET = """
You are an intelligent agent equipped with an RGB sensor, object detector, and Visual Question Answering (VQA) model.
Your goal is to identify a target {target_object} based on a scene description and prior knowledge of the target.

Scenario:
The object detector has identified a scene containing a {target_object}, and the VQA model has provided the following description:

<START_OF_DESCRIPTION>
{distractor_object_description}
<END_OF_DESCRIPTION>

Target object information: 
Based on previous interactions, you know the target picture has the following characteristics:
<START_TARGET_PICTURE_FACTS>
{facts_about_the_target_picture}
<END_TARGET_PICTURE_FACTS> 

Task:
1. Similarity analysis.
Analyze how closely the detected scene description aligns with the known facts about the target {target_object}. Provide a similarity score between 0 and 10, where:
- 0 = The detected {target_object} is not the target object.
- 10 = The detected {target_object} is definitely the target object.
- If no information about the target is available, the score should be -1.

2. Question Generation:
- The question is for the target object, not the detected one.
- Ask exactly one specific, relevant, and human-answerable question related to the target object that maximizes information gain for identifying the target {target_object}.
- Do not ask speculative or irrelevant questions 
- The question should be grounded in observable or known details from the scene, focusing on key characteristics that can help confirm or refute the identity of the target object.

Ensure your response follows the format below:
YAML_START # must be present to get the information back
similarity_score: <similarity score>
questions:
    <question_number>: <question_content>
YAML_END # must be present to get the information back

Provide your reasoning step-by-step for the similarity score and questions, after the YAML_END tag."""

################################################
#  Prompt sentences
################################################

_TASK_PROMPT = "You are a robot navigating an enclosed space. Your goal is to navigate to the correct object based on the user's commands. \
You were given the following task by the user '{USER_TASK}'. Currently, you are facing a scene represented by the given image. "

_TASK_PROMPT_WITH_CLASS = _TASK_PROMPT[:-2] + ", which contains a {OBJCLASS}. "

_REASONING_PROMPT_WITH_CLASS = (
    "Reason about the {OBJCLASS} that you are seeing in the image, comparing what you know about the task \
(given the user commands) and the given scene. For example, \
if the task is 'Navigate to the black leather sofa near a lampstand' your reasoning process will be 'I'm currently observing a brown sofa \
which is different than black, making it unlikely to be the target sofa. Moreover, there is no lampstand near it, only a rug and a window' \
etc. If there are distortions or artifact, do not focus on them, focus on the object at hand. "
)

_REASONING_PROMPT = (
    "Reason about what you are seeing, comparing what you know about the task (given the user commands) and the given scene. \
For example, if the task is 'Navigate to the black leather sofa near a lampstand' your reasoning process will be 'I'm currently observing a brown sofa \
which is different than black, making it unlikely to be the target sofa. Moreover, there is no lampstand near it, only a rug and a window' \
etc. If there are distortions or artifact, do not focus on them, focus on the object at hand. "
)

_SCORE_PROMPT = "Return a score from 1 to 5 on how much you are sure about the fact that the current observation \
(given by the image) matches the target task given by the user, where 1 means 'extremely unlikely' and 5 'extremely likely'. Be concise \
(use at most 100 words) and return a response with the following form: <score>Your score</score>"

_SCORE_PROMPT_WITH_REASONING = "At the end of the reasoning process, return a score from 1 to 5 on how much you are sure about the fact that the current observation \
(given by the image) matches the target task given by the user, where 1 means 'extremely unlikely' and 5 'extremely likely'. Be concise \
(use at most 100 words) and return a response with the following form: <motivation>Your motivations</motivation><score>Your score</score>"

_CHOICE_PROMPT = (
    "Evaluate how well the provided image aligns with the user's task. Assign a confidence score based on the following scale: \
- 0: You are certain the image **DOES NOT** match the task. \
- 1: You are unsure whether the image matches the task or not. \
- 2: You are certain the image **DOES** match the task. \
Strictly follow this output format: \
<score>0, 1, or 2</score>"
)

_CHOICE_PROMPT_WITH_REASONING = (
    "At the end of the reasoning process, evaluate how well the provided image aligns with the user's task. Assign a confidence score based on the following scale: \
- 0: You are certain the image **DOES NOT** match the task. \
- 1: You are unsure whether the image matches the task or not. \
- 2: You are certain the image **DOES** match the task. \
Provide a concise reasoning (under 100 words) and strictly follow this output format: \
<motivation>Your reasoning here</motivation><score>0, 1, or 2</score>"
)


################################################
#  Actual prompts
################################################

# PROMPT_WITH_CLASS = (
#     "You are a robot navigating an enclosed space. Your goal is to navigate to the correct object based on the user's commands.\
# You were given the following task by the user '{USER_TASK}'. Currently, you are facing a scene represented by the given image, which contains a {OBJCLASS}. \
# Reason about the {OBJCLASS} that you are seeing in the image, comparing what you know about the task (given the user commands) and the given scene. For example, \
# if the task is 'Navigate to the black leather sofa near a lampstand' your reasoning process will be 'I'm currently observing a brown sofa \
# which is different than black, making it unlikely to be the target sofa. Moreover, there is no lampstand near it, only a rug and a window' \
# etc. If there are distortions or artifact, do not focus on them, focus on the object at hand. At the end of the reasoning process, give it a score from 1 to 5 on how much \
# you are sure about the fact that the current observation \ (given by the image) matches the target task given by the user, \
# where 1 means 'extremely unlikely' and 5 'extremely likely'. Be concise (use at most 100 words) and return a response with \
#  the following form: <motivation>Your motivations</motivation><score>Your score</score>"
# )

SINGLEMODEL_PROMPT_WITH_CLASS = _TASK_PROMPT_WITH_CLASS + _REASONING_PROMPT_WITH_CLASS + _SCORE_PROMPT_WITH_REASONING

# PROMPT_WITH_CLASS_WITH_CHOICES = (
#     "You are a robot navigating an enclosed space. Your goal is to navigate to the correct object based on the user's commands. \
# You were given the following task by the user '{USER_TASK}'. Currently, you are facing a scene represented by the given image, which contains a {OBJCLASS}. \
# Reason about the {OBJCLASS} that you are seeing in the image, comparing what you know about the task (given the user commands) and the given scene. For example, \
# if the task is 'Navigate to the black leather sofa near a lampstand' your reasoning process will be 'I'm currently observing a brown sofa \
# which is different than black, making it unlikely to be the target sofa. Moreover, there is no lampstand near it, only a rug and a window' \
# etc. If there are distortions or artifact, do not focus on them, focus on the object at hand. At the end of the reasoning process, return one among \
# three possible answers: 2 if you are sure the current observation matches the target task given by the user, 1 if you are unsure whether the image \
# matches the task or not, 0 if you are sure that the image DOES NOT match the task. Return a response with the following form: \
# <motivation>Your motivations</motivation><score>Your Answer</score>"
# )
SINGLEMODEL_PROMPT_WITH_CLASS_WITH_CHOICES = (
    _TASK_PROMPT_WITH_CLASS + _REASONING_PROMPT_WITH_CLASS + _CHOICE_PROMPT_WITH_REASONING
)

# PROMPT_WITH_CLASS_ONLY_SCORE = (
#     "You are a robot navigating an enclosed space. Your goal is to navigate to the correct object based on the user's commands. \
# You were given the following task by the user '{USER_TASK}'. Currently, you are facing a scene represented by the given image, which contains a {OBJCLASS}. \
# If there are distortions or artifact, do not focus on them, focus on the object at hand. Give it a score from 1 to 5 on how much you are sure about the fact that the current observation \
# (given by the image) matches the target task given by the user, where 1 means 'extremely unlikely' and 5 'extremely likely'. Return a response with \
# the following form: <score>Your score</score>"
# )
SINGLEMODEL_PROMPT_WITH_CLASS_ONLY_SCORE = _TASK_PROMPT_WITH_CLASS + _SCORE_PROMPT

# PROMPT_ONLY_SCORE = (
#     "You are a robot navigating an enclosed space. Your goal is to navigate to the correct object based on the user's commands. \
# You were given the following task by the user '{USER_TASK}'. Currently, you are facing a scene represented by the given image. \
# If there are distortions or artifact, do not focus on them, focus on the object at hand. Give it a score from 1 to 5 on how much you are sure about the fact that the current observation \
# (given by the image) matches the target task given by the user, where 1 means 'extremely unlikely' and 5 'extremely likely'. Return a response with \
# the following form: <score>Your score</score>"
# )
SINGLEMODEL_PROMPT_ONLY_SCORE = _TASK_PROMPT + _SCORE_PROMPT

# PROMPT_ONLY_CHOICES = (
#     "You are a robot navigating an enclosed space. Your goal is to navigate to the correct object based on the user's commands. \
# You were given the following task by the user '{USER_TASK}'. Currently, you are facing a scene represented by the given image. \
# If there are distortions or artifact, do not focus on them, focus on the object at hand. Return one among \
# three possible answers: 2 if you are sure the current observation matches the target task given by the user, 1 if you are unsure whether the image \
# matches the task or not, 0 if you are sure that the image DOES NOT match the task. Return a response with the following form: \
# <motivation>Your motivations</motivation><score>Your Answer</score>"
# )
SINGLEMODEL_PROMPT_ONLY_CHOICES = _TASK_PROMPT + _CHOICE_PROMPT

# PROMPT = (
#     "You are a robot navigating an enclosed space. Your goal is to navigate to the correct object based on the user's commands.\
# You were given the following task by the user '{USER_TASK}'. Currently, you are facing a scene represented by the given image. \
# Reason about what you are seeing, comparing what you know about the task (given the user commands) and the given scene. For example, \
# if the task is 'Navigate to the black leather sofa near a lampstand' your reasoning process will be 'I'm currently observing a brown sofa \
# which is different than black, making it unlikely to be the target sofa. Moreover, there is no lampstand near it, only a rug and a window' \
# etc. If there are distortions or artifact, do not focus on them, focus on the object at hand. At the end of the reasoning process, give it a score from 1 to 5 on how much you are sure about the fact that the current observation \
# (given by the image) matches the target task given by the user, where 1 means 'extremely unlikely' and 5 'extremely likely'. Be concise (use at most 100 words) and return a response with \
#  the following form: <motivation>Your motivations</motivation><score>Your score</score>"
# )
SINGLEMODEL_PROMPT_SCORE = _TASK_PROMPT + _REASONING_PROMPT + _SCORE_PROMPT_WITH_REASONING

# PROMPT_CHOICES = (
#     "You are a robot navigating an enclosed space. Your goal is to navigate to the correct object based on the user's commands. \
# You were given the following task by the user '{USER_TASK}'. Currently, you are facing a scene represented by the given image. \
# Reason about what you are seeing, comparing what you know about the task (given the user commands) and the given scene. For example, \
# if the task is 'Navigate to the black leather sofa near a lampstand' your reasoning process will be 'I'm currently observing a brown sofa \
# which is different than black, making it unlikely to be the target sofa. Moreover, there is no lampstand near it, only a rug and a window' \
# etc. If there are distortions or artifact, do not focus on them, focus on the object at hand. At the end of the reasoning process, return one among \
# three possible answers: 2 if you are sure the current observation matches the target task given by the user, 1 if you are unsure whether the image \
# matches the task or not, 0 if you are sure that the image DOES NOT match the task. Return a response with the following form: \
# <motivation>Your motivations</motivation><score>Your Answer</score>"
# )
SINGLEMODEL_PROMPT_CHOICES = _TASK_PROMPT + _REASONING_PROMPT + _CHOICE_PROMPT_WITH_REASONING

SINGLEMODEL_PROMPT_ASK_FOR_PROPERTIES = "Currently, you are facing a scene represented by the given image, which contains a {OBJCLASS}. If there are distortions or artifact, do not focus on them, focus on the object at hand. \
Answer these three questions: \
    1. What color is the {OBJCLASS} that you are seeing? 2. What shape/dimensions does it have? 3. What other objects are near? \
    Return an asnwer formatted like this <color>color answer</color>\n<shape>shape and dimension answer</shape>\n<objects>objects answer</objects>\n"

SINGLE_MODEL_GOTO_PROMPT = "Go to the {OBJECT_DESC}"