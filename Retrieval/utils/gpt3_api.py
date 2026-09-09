import openai


def get_gpt_3_response_local(prompt, engine, temperature, stop):
    openai.api_key = "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"  # API
    # response = openai.Completion.create(
    #     engine=engine,
    #     prompt=prompt,
    #     max_tokens=64,
    #     temperature=temperature,
    #     stop=stop,
    # )

    response = openai.ChatCompletion.create(
        model=engine,
        messages=[
            {"role": "system", "content": "You are a text completion assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=64,
        temperature=temperature,
        stop=stop,
        n=1,
    )
    return response

def get_gpt_3_response(prompt, engine, temperature, stop):
    openai.api_key = "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"  # API
    # openai.api_key = "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"  # API
    # response = openai.Completion.create(
    #     engine=engine,
    #     prompt=prompt,
    #     max_tokens=64,
    #     temperature=temperature,
    #     stop=stop,
    # )

    response = openai.ChatCompletion.create(
        model=engine,
        messages=[
            {"role": "system", "content": "You are a text completion assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=128,
        temperature=temperature,
        stop=stop,
        n=1,
    )
    return response

if __name__ == '__main__':

    prompt='Instruction：你是一个中国网友，请以所给话题为主题发表自己的观点性评论。输出长度限制在64字以内。' \
           'Topic：#就台湾地区对大陆贸易限制措施进行贸易壁垒调查的公告#' \
           'Generated："是新一轮的起手式！<EOG>"' \
           'Topic：#就台湾地区对大陆贸易限制措施进行贸易壁垒调查的公告#' \
           'Generated："这可是好事啊！采购他们的材料还要先付款再发货！出了品质问题还没办法罚他们款！ <EOG>"'\
           'Topic：#就台湾地区对大陆贸易限制措施进行贸易壁垒调查的公告#' \
           'Generated："挤压台独的生存空间，就是扩大统一的完成空间。要积极统一！ <EOG>"' \
           'Topic：#小伙年会喜提365天带薪休假#' \
           'Generated: '
    response=get_gpt_3_response(prompt=prompt, engine='gpt-3.5-turbo', temperature=1, stop='<EOG>')
    print(response["choices"][0]["message"]["content"])