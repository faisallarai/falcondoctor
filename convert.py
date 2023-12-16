import pandas as pd
import json

# read json
with open('./data/alpaca_data.json', encoding='utf-8') as f:
    alpaca_data = json.loads(f.read())
    
with open('./data/GenMedGPT-5k.json', encoding='utf-8') as f:
    genmedgpt_data = json.loads(f.read())
    
with open('./data/HealthCareMagic-100k.json', encoding='utf-8') as f:
    healthcaremagic = json.loads(f.read())

df = pd.json_normalize(alpaca_data)
df = pd.json_normalize(genmedgpt_data)
df = pd.json_normalize(healthcaremagic)


df.to_csv('alpaca_data.csv', index=False, encoding='utf-8')
df.to_csv('genmedgpt-5k.csv', index=False, encoding='utf-8')
df.to_csv('healthcaremagic-100k.csv', index=False, encoding='utf-8')

