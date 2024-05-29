#!/usr/bin/env python
# encoding: utf-8
# @author: yihuai lan
# @fileName: run_avalon_chatglm_agent.py
# @date: 2024/4/11 11:27
#
# describe:
#
# !/usr/bin/env python
# encoding: utf-8
# @author: yihuai lan
# @fileName: run_avalon_battle.py
# @date: 2024/3/6 17:46
#
# describe:
#
import random
import argparse

import openai
import torch.nn
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, AutoModel

from src.extractor.llm_extractor.chatgpt_extractor import ChatGPTBasedExtractor
from src.games.avalon.avalon import Avalon
# from src.games.werewolf.werewolf import Werewolf
from src.agents import ChatGPT_SAPARAgent_ForAvalon, ChatGPT_CGAgent, ChatGLM_GraphCGAgent, ChatGLM_CodeActAgent, ChatGLM_CodeActAgent_forWerewolf, ChatGLM_CodeActAgent_forAvalon
# from prompt.avalon_sapar_prompt import summary_prompt, plan_prompt, response_prompt, system_prompt, \
#     action_prompt, suggestion_prompt, update_prompt, analysis_prompt, \
#     strategy_prompt, candidate_actions, init_strategies, role_introduction, role_target
# from prompt.avalon_graph_cg_prompt import rule_role_prompt, select_question_prompt, ask_question_prompt, generate_answer_prompt, reflection_prompt, \
#     extract_suggestion_prompt, generate_response_prompt, informativeness_prompt,
from prompt.avalon_codeact_prompt import rule_role_prompt, select_question_prompt, ask_question_prompt, \
    generate_answer_prompt, reflection_prompt, extract_suggestion_prompt, generate_response_prompt, \
    informativeness_prompt, question_list, code_generate_prompt, generate_team_leader_prompt
from prompt.avalon_graph_cg_agent_prompt import graph_generate_prompt, reasoning_prompt

# from prompt.werewolf_codeact_prompt import rule_role_prompt, select_question_prompt, ask_question_prompt, \
#     generate_answer_prompt, reflection_prompt, extract_suggestion_prompt, generate_response_prompt, \
#     informativeness_prompt, question_list, code_generate_prompt, generate_team_leader_prompt
from src.games.avalon.extract_demos import number_extract_prompt, player_extractor_demos, vote_extractor_demos, \
    quest_extractor_demos, choose_identify_extractor_demos, select_merlin_extractor_demos, bool_extract_prompt, \
    quest_extract_prompt
# from src.games.werewolf.extract_demos import number_extract_prompt, player_extractor_demos, vote_extractor_demos, \
#     bool_extract_prompt
from src.utils import create_dir, read_json

api_key = "sk-FyPQ0uwONAZnfD5I17C3A8C03cA04f64900887B0Ef9c0022"
base_url = None
roles = ["Merlin", "Percival", "Loyal Servant", "Loyal Servant", "Morgana", "Assassin"]
# roles = ["Werewolf", "Werewolf", "Villager", "Villager", "Seer", "Guard", "Witch"]


bert_model = SentenceTransformer("/home/hdd/lijinyi/models/sentence-transformers/multi-qa-mpnet-base-cos-v1", device="cuda", trust_remote_code=True)

# load chatglm
abs_model_dir = "/home/hdd/lijinyi/models/THUDM/chatglm3-6b-32k"
tokenizer = AutoTokenizer.from_pretrained(abs_model_dir, trust_remote_code=True)
model = AutoModel.from_pretrained(abs_model_dir, trust_remote_code=True).half().to("cuda:0")
model = model.eval()


def load_llama_model(llama_dir):
    print(f"llama model loading: {llama_dir}")
    config = AutoConfig.from_pretrained(llama_dir, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(llama_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(llama_dir,
                                                 torch_dtype='auto', ignore_mismatched_sizes=True,
                                                 trust_remote_code=True, config=config, local_files_only=True).half().to("cuda:1")
    model = model.eval()
    tokenizer.pad_token = tokenizer.eos_token
    print(f"loading finish")
    return model, tokenizer, config


model_name = "meta-llama/Llama-2-7b-chat-hf"
abs_llama_model_dir = f"/home/hdd/lijinyi/models/{model_name}"
llama_model, llama_tokenizer, _ = load_llama_model(abs_llama_model_dir)


def run_game(game_output_dir: str, camp, game_idx):
    create_dir(game_output_dir.format(game_idx))

    mode = 'watch'
    language = 'english'
    ai_model = 'gpt-3.5-turbo-16k'
    player_nums = 6
    player_mapping = {}
    random.shuffle(roles)
    if camp == "good":
        camp_role = ["Merlin", "Percival", "Loyal Servant"]
    else:
        camp_role = ["Morgana", "Assassin"]
    # camp_role = ["Merlin", "Percival", "Loyal Servant", "Morgana", "Assassin"]
    # if camp == "villager":
    #     camp_role = ["Villager", "Seer", "Guard", "Witch"]
    # else:
    #     camp_role = ["Werewolf"]
    # camp_role = ["Werewolf", "Villager", "Seer", "Guard", "Witch"]

    game = Avalon(player_nums, language, mode, ai_model, game_output_dir.format(game_idx))
    # game = Werewolf(player_nums, language, mode, ai_model, game_output_dir.format(game_idx))

    player_args = []
    for i in range(player_nums):
        log_dir = f"{game_output_dir.format(game_idx)}/player {i + 1}"
        create_dir(log_dir)
        if roles[i] in camp_role:
            name = f"player {i + 1}"
            role = roles[i]

            player_args.append(
                (
                    # ChatGPT_CGAgent, {"name": name, "role": role, "rule_role_prompt": rule_role_prompt,
                    #                   "select_question_prompt": select_question_prompt,
                    #                   "ask_question_prompt": ask_question_prompt,
                    #                   "generate_answer_prompt": generate_answer_prompt,
                    #                   "reflection_prompt": reflection_prompt,
                    #                   "extract_suggestion_prompt": extract_suggestion_prompt,
                    #                   "generate_response_prompt": generate_response_prompt,
                    #                   "informativeness_prompt": informativeness_prompt,
                    #                   "question_list": question_list.get(roles[i], []), "retrival_model": bert_model,
                    #                   "model": model, "freshness_k": 15, "informativeness_n": 15,
                    #                   "experience_window": 50, "tokenizer": tokenizer,
                    #                   "temperature": 0.3, "api_key": "", "previous_exp_pool": previous_exp_pool,
                    #                   "output_dir": log_dir}
                    ChatGLM_CodeActAgent_forAvalon, {"name": name, "role": role, "rule_role_prompt": rule_role_prompt,
                                           "private_information": "", "current_team_number": 2,
                                           "code_generate_prompt": code_generate_prompt, "total_player_number": player_nums,
                                           "good_number": 4, "bad_number": 2,
                                           "generate_leader_response_prompt": generate_team_leader_prompt,
                                           "select_question_prompt": select_question_prompt,
                                           "ask_question_prompt": ask_question_prompt,
                                           "generate_answer_prompt": generate_answer_prompt,
                                           "reflection_prompt": reflection_prompt,
                                           "extract_suggestion_prompt": extract_suggestion_prompt,
                                           "generate_response_prompt": generate_response_prompt,
                                           "informativeness_prompt": informativeness_prompt,
                                           "question_list": question_list.get(roles[i], []), "retrieval_model": bert_model,
                                           "model": model, "tokenizer": tokenizer,
                                           "llama_model": llama_model, "llama_tokenizer": llama_tokenizer,
                                           "k": 15, "informativeness_n": 15,
                                           "temperature": 0.3, "api_key": api_key, "output_dir": log_dir}
                )
            )
            player_mapping[name] = role
        else:
            if game_idx == 0:
                previous_exp_pool = []
            else:
                load_file = f"{game_output_dir.format(game_idx - 1)}/{roles[i]}_reflection.json"
                previous_exp_pool = read_json(load_file)
            name = f"player {i + 1}"
            role = roles[i]
            player_args.append(
                (
                    # ChatGPT_CGAgent, {"name": name, "role": role, "rule_role_prompt": rule_role_prompt,
                    #                   "select_question_prompt": select_question_prompt,
                    #                   "ask_question_prompt": ask_question_prompt,
                    #                   "generate_answer_prompt": generate_answer_prompt,
                    #                   "reflection_prompt": reflection_prompt,
                    #                   "extract_suggestion_prompt": extract_suggestion_prompt,
                    #                   "generate_response_prompt": generate_response_prompt,
                    #                   "informativeness_prompt": informativeness_prompt,
                    #                   "question_list": question_list.get(roles[i], []), "retrival_model": bert_model,
                    #                   "model": model, "freshness_k": 15, "informativeness_n": 15,
                    #                   "experience_window": 50, "tokenizer": tokenizer,
                    #                   "temperature": 0.3, "api_key": "", "previous_exp_pool": previous_exp_pool,
                    #                   "output_dir": log_dir}
                    ChatGLM_GraphCGAgent, {"name": name, "role": role, "rule_role_prompt": rule_role_prompt,
                                           "graph_generate_prompt": graph_generate_prompt, "reasoning_prompt": reasoning_prompt,
                                           "select_question_prompt": select_question_prompt,
                                           "ask_question_prompt": ask_question_prompt,
                                           "generate_answer_prompt": generate_answer_prompt,
                                           "reflection_prompt": reflection_prompt,
                                           "extract_suggestion_prompt": extract_suggestion_prompt,
                                           "generate_response_prompt": generate_response_prompt,
                                           "informativeness_prompt": informativeness_prompt,
                                           "question_list": question_list.get(roles[i], []),
                                           "retrival_model": bert_model,
                                           "freshness_k": 10,
                                           "model": model, "tokenizer": tokenizer,
                                           "experience_window": 30, "previous_exp_pool": previous_exp_pool,
                                           "informativeness_n": 10,
                                           "temperature": 0.3, "api_key": api_key, "output_dir": log_dir}
                )
            )


    game.add_players(player_args)

    # Avalon extractors
    extractor_args = [(ChatGPTBasedExtractor,
                       {"extractor_name": "player extractor", "model_name": 'gpt-3.5-turbo-16k',
                        "extract_prompt": number_extract_prompt, "system_prompt": "You are a helpful assistant.",
                        "temperature": 0, "few_shot_demos": player_extractor_demos,
                        "output_dir": game_output_dir.format(game_idx)}),
                      (ChatGPTBasedExtractor,
                       {"extractor_name": "vote extractor", "model_name": 'gpt-3.5-turbo-16k',
                        "extract_prompt": bool_extract_prompt, "system_prompt": "You are a helpful assistant.",
                        "temperature": 0, "few_shot_demos": vote_extractor_demos,
                        "output_dir": game_output_dir.format(game_idx)}),
                      (ChatGPTBasedExtractor,
                       {"extractor_name": "quest extractor", "model_name": 'gpt-3.5-turbo-16k',
                        "extract_prompt": quest_extract_prompt, "system_prompt": "You are a helpful assistant.",
                        "temperature": 0, "few_shot_demos": quest_extractor_demos,
                        "output_dir": game_output_dir.format(game_idx)}),
                      (ChatGPTBasedExtractor,
                       {"extractor_name": "identify extractor", "model_name": 'gpt-3.5-turbo-16k',
                        "extract_prompt": bool_extract_prompt, "system_prompt": "You are a helpful assistant.",
                        "temperature": 0, "few_shot_demos": choose_identify_extractor_demos,
                        "output_dir": game_output_dir.format(game_idx)}),
                      (ChatGPTBasedExtractor,
                       {"extractor_name": "merlin extractor", "model_name": 'gpt-3.5-turbo-16k',
                        "extract_prompt": number_extract_prompt, "system_prompt": "You are a helpful assistant.",
                        "temperature": 0, "few_shot_demos": select_merlin_extractor_demos,
                        "output_dir": game_output_dir.format(game_idx)})]

    # Werewolf extractors
    # extractor_args = [(ChatGPTBasedExtractor,
    #                    {"extractor_name": "player extractor", "model_name": 'gpt-3.5-turbo-16k',
    #                     "extract_prompt": number_extract_prompt, "system_prompt": "You are a helpful assistant.",
    #                     "temperature": 0, "few_shot_demos": player_extractor_demos,
    #                     "output_dir": game_output_dir.format(game_idx)}),
    #                   (ChatGPTBasedExtractor,
    #                    {"extractor_name": "antidote extractor", "model_name": 'gpt-3.5-turbo-16k',
    #                     "extract_prompt": bool_extract_prompt, "system_prompt": "You are a helpful assistant.",
    #                     "temperature": 0, "few_shot_demos": vote_extractor_demos,
    #                     "output_dir": game_output_dir.format(game_idx)}),
    #                   (ChatGPTBasedExtractor,
    #                    {"extractor_name": "killing agreement extractor", "model_name": 'gpt-3.5-turbo-16k',
    #                     "extract_prompt": bool_extract_prompt, "system_prompt": "You are a helpful assistant.",
    #                     "temperature": 0, "few_shot_demos": vote_extractor_demos,
    #                     "output_dir": game_output_dir.format(game_idx)})
    #                   ]

    game.init_extractor(player_extractor=extractor_args[0], vote_extractor=extractor_args[1],
                        quest_extractor=extractor_args[2],
                        choose_identify_extractor=extractor_args[3], select_merlin_extractor=extractor_args[4])

    # game.init_extractor(player_extractor=extractor_args[0], antidote_extractor=extractor_args[1],
    #                     killing_agreement_extractor=extractor_args[2])

    game.start()
    for player_i, agent_i in game.players.items():
        if isinstance(agent_i, (ChatGPT_SAPARAgent_ForAvalon, ChatGPT_CGAgent, ChatGLM_GraphCGAgent)):
            agent_i.reflection(
                player_mapping,
                file_name=f"{game_output_dir.format(game_idx)}/{player_mapping.get(player_i)}_reflection.json",
                winners=game.winners,
                duration=game.game_round
                # duration=game.day_count
            )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--game_count", type=int, default=10
    )
    parser.add_argument("--camp", type=str, default="good", choices=["good", "evil"])
    parser.add_argument("--exp_name", type=str, default="battle")
    # parser.add_argument("--camp", type=str, default="villager", choices=["villager", "werewolf"])

    parser.add_argument("--use_proxy", type=bool, default=False)
    parser.add_argument("--start_game_idx", type=int, default=0)
    parsed_args = parser.parse_args()
    return parsed_args


def main():
    args = parse_args()
    # if args.use_proxy:
    #     openai.proxy = "http://127.0.0.1:7890"
    #     os.environ["ALL_PROXY"] = "http://127.0.0.1:7890"

    openai.api_key = api_key
    openai.base_url = base_url
    for game_round in range(args.start_game_idx, args.game_count):
        output_dir = f"playing_log/avalon/battle/{args.exp_name}-{args.camp}" + "-game_{}"
        run_game(output_dir, camp=args.camp, game_idx=game_round)
        print("game finish!!! game index {}".format(game_round))


if __name__ == '__main__':
    main()
    print("done!!!")