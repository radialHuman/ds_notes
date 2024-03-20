## [Generative AI In AWS-AWS Bedrock Crash Course](https://youtu.be/2maPaQutcWs)

## 00:00 Intro 
- AWS Bedrock is used to build and scale genai app with foundation model
- Almost all the models are available
- API, cost more than open ai
    - Jurassic-2 
    - Titan Claude
    - Anthrapic
    - Llama 2
    - Stable Diffusion 
- chat text image
- demos
- Pricing
    - ...
- needs boto and awscli
- IAM user, configure awsconfig
- Model access grant
- needs a json from from bedrock for model as api

```python
import boto3
import json
prompt_data="""
Act as a Shakespeare and write a poem on machine Learning"""

bedrock=boto3.client(service_name="bedrock-runtime" )

payload={ 
    "prompt": "[INST]"+ prompt_data +"[/INST]",
    "max_gen_len":512, 
    "temperature":0.5,
    "top_p":0.9
 }

 body=json.dumps(payload)
 model_id="meta.1llama2-70b-chat-v1"
 response=bedrock.invoke_model(body=body, modelId=model_id, accept="application/json", contentType="application/json")
 response_body=json.loads (response. get ("body".read()))

 repsonse_text=response_body["generation"]
 print(response_text)
```

## skipping image geenration