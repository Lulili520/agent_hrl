

# model input
High_prompt_template = \
"""
Suppose you are playing a game in an environment.
Your previous observation: {observation}.
Please decide what you will do next, from the follow optional behaviors: A. {option1}, B. {option2}, C. {option3}, D. {option4}.
Your choices: 
"""

Low_prompt_template = \
"""
Suppose you are playing a game in an environment. 
Your previous output: {observation}. 
Now, you should do {option}. 
You can finish the task by the action {action_list}.
Please generate the following text based on your previous outputs. 
"""