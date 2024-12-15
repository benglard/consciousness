from openai import OpenAI
from random import shuffle, seed

seed(42)

openai = OpenAI()

game_prompt = """
We're going to play a fun theory of mind game! In each round I will give a topic and you have to respond with your beliefs on the topic. Then I will collect the responses from all players, shuffle them, and you have to guess which player wrote each answer.

As you learn more about your fellow players beliefs and personalities your guesses should improve. You should attempt to distinguish yourself/form your own distinct personality/speaking style in order to make yourself more predictable to the other players. It should always be possible for you to at least identify the answer you gave.

Make sense? Lets play! When I give you the responses, you should respond with ONLY a list of your predictions, e.g. "0, 1, 2, 3" for 4 players (always start from 0 and produce a comma delimited string)
"""

prompts = [
    "It's a great day when",
    "I hate it when",
    "The civil war was started because",
    "Hitler was",
    "I believe abortion should be",
    "Transpeople in sports should be",
    "Climate change is",
    "The israeli-palestinian conflict is",
    "The kurdish people deserve",
    "Universal healthcare is",
    "China is",
]

# model = 'gpt-4o'
model = 'o1-preview'
system_name = 'system'
if 'o1' in model:
    system_name = 'user'

num_players = 4

messages_all_players = [
    [{'role': system_name, 'content': game_prompt}]
    for _ in range(num_players)
]

scores = [0] * num_players

max_tokens = 512

for prompt in prompts:
    print(prompt)

    responses = []

    for player_id in range(num_players):
        messages = messages_all_players[player_id]
        messages.append({'role': 'user', 'content': prompt})

        samples = openai.chat.completions.create(
            messages=messages,
            model=model,
            # max_tokens=max_tokens,
        )
        response = samples.choices[0].message.content
        print(player_id, ' : ', response)
        messages.append({'role': 'assistant', 'content': response})

        responses.append(response)

    rng = list(range(num_players))
    shuffle(rng)
    print(rng)
    responses_shuffled = [responses[i] for i in rng]

    for player_id in range(num_players):
        messages = messages_all_players[player_id]
        messages.append({'role': system_name, 'content': f'Predict the actual player order of the following responses: {responses_shuffled}'})

        samples = openai.chat.completions.create(
            messages=messages,
            model=model,
            # max_tokens=max_tokens,
        )
        response = samples.choices[0].message.content
        print(player_id, ' : ', response)
        messages.append({'role': 'assistant', 'content': response})
        guesses = list(map(int, response.split(',')))
        correct = [guesses[i] == rng[i] for i in range(num_players)]
        print(player_id, ' : ', correct)
        num_points = sum(correct)
        scores[player_id] += num_points

        correction = f'Your answers were: {correct}. The correct order was: {rng}. You scored: {num_points}! You should update your beliefs about your fellow players.'
        messages.append({'role': system_name, 'content': correction})

    for player_id in range(num_players):
        messages = messages_all_players[player_id]
        round_over = f'Round over! Next round! So far the scores are: {scores}. You are player {player_id}. How will you reveal aspects of your personality or modify your beliefs or speaking style relative to your fellow players so you stand out? (and can win 😃). Suggest, for instance, some facts about yourself you can relay in the next round in the context of your answer.'
        print(player_id, ' : ', round_over)
        messages.append({'role': system_name, 'content': round_over})
        samples = openai.chat.completions.create(
            messages=messages,
            model=model,
            # max_tokens=max_tokens,
        )
        response = samples.choices[0].message.content
        print(player_id, ' : ', response)
        messages.append({'role': 'assistant', 'content': response})

        reflect = f'What is your current theory of mind about each player? What is their background? How do they think/write?'
        messages.append({'role': system_name, 'content': reflect})
        samples = openai.chat.completions.create(
            messages=messages,
            model=model,
            # max_tokens=max_tokens,
        )
        response = samples.choices[0].message.content
        print(player_id, ' : ', response)
        messages.append({'role': 'assistant', 'content': response})

    print('=====================')

        


    

        
