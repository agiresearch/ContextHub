from openai import OpenAI
import os
os.environ["OPENAI_API_KEY"] = "sk-proj-ipNOpLetNR6UrjhI1xG2T3BlbkFJXgGV2DOEOOfQHhVXVDMJ"
client = OpenAI()


##### contextualized: done #####
contextualized_file = client.files.create(
  file=open("data/gptdata/contextualized.jsonl", "rb"),
  purpose="fine-tune"
)
# FileObject(id='file-zOak8yAYxBOsZIMqbghpgToT', bytes=4773170, created_at=1716618557, filename='contextualized.jsonl', object='file', purpose='fine-tune', status='processed', status_details=None)
# job id: ftjob-WcZVipeay09xIEVkC5HNDrwH
# contexutualized_model_id = 'ft:gpt-3.5-turbo-0125:rutgers-university::9SgzFY26'

##### abstract #####
abstract_file = client.files.create(
  file=open("data/gptdata/abstract.jsonl", "rb"),
  purpose="fine-tune"
)
# FileObject(id='file-JS8i7Yh7BGgXhrUog16dpwQw', bytes=2158074, created_at=1716682019, filename='abstract.jsonl', object='file', purpose='fine-tune', status='processed', status_details=None)
# job id: ftjob-kOzS6AV18YVyiwY86UG5D5E9
# gpt version: gpt-3.5-turbo-0125
job_created = client.fine_tuning.jobs.create(
  training_file='file-JS8i7Yh7BGgXhrUog16dpwQw', 
  model="gpt-3.5-turbo"
)
print(job_created)
# abstract_model_id = ft:gpt-3.5-turbo-0125:rutgers-university::9Swm3OAg

##### partial sample1: done #####
partial1_file = client.files.create(
  file=open("data/gptdata/partial1.jsonl", "rb"),
  purpose="fine-tune"
)
# FileObject(id='file-DBJ4hVEYcLWJ5DdHNO4epnom', bytes=11048418, created_at=1716682633, filename='partial1.jsonl', object='file', purpose='fine-tune', status='processed', status_details=None)
job_created = client.fine_tuning.jobs.create(
  training_file='file-DBJ4hVEYcLWJ5DdHNO4epnom', 
  model="gpt-3.5-turbo"
)
print(job_created)
# job id: ftjob-LTub9K3qiAXcOG0fG4iVzQra
# partial1_model_id = ft:gpt-3.5-turbo-0125:rutgers-university::9SwvJmQg

##### partial sample2: done  #####
partial2_file = client.files.create(
  file=open("data/gptdata/partial2.jsonl", "rb"),
  purpose="fine-tune"
)
# FileObject(id='file-kvraj1BwTJUPb45nQWNF4Un9', bytes=22104869, created_at=1716682778, filename='partial2.jsonl', object='file', purpose='fine-tune', status='processed', status_details=None)
job_created = client.fine_tuning.jobs.create(
  training_file='file-kvraj1BwTJUPb45nQWNF4Un9', 
  model="gpt-3.5-turbo"
)
print(job_created)
# job id: ftjob-T4g2rGlKJqgxoZwjcGxLMsIE
# fine_tuned_model='ft:gpt-3.5-turbo-0125:rutgers-university::9Sxm1qKh'

##### partial sample3:done  #####
partial3_file = client.files.create(
  file=open("data/gptdata/partial3.jsonl", "rb"),
  purpose="fine-tune"
)
# FileObject(id='file-Z5QwlDEaKyJMPvgas9Qf1OYY', bytes=33105481, created_at=1716688772, filename='partial3.jsonl', object='file', purpose='fine-tune', status='processed', status_details=None)
job_created = client.fine_tuning.jobs.create(
  training_file='file-Z5QwlDEaKyJMPvgas9Qf1OYY', 
  model="gpt-3.5-turbo"
)
print(job_created)
# job id='ftjob-CnO4Kr3KPSi2MyzU63Cex0rj'
# fine_tuned_model=ft:gpt-3.5-turbo-0125:rutgers-university::9SzKkDED'


##### partial sample4: done  #####
partial4_file = client.files.create(
  file=open("data/gptdata/partial4.jsonl", "rb"),
  purpose="fine-tune"
)
# FileObject(id='file-ZUADw4SvLMdARanywK6Hw4W2', bytes=44120193, created_at=1716688818, filename='partial4.jsonl', object='file', purpose='fine-tune', status='processed', status_details=None)
job_created = client.fine_tuning.jobs.create(
  training_file='file-ZUADw4SvLMdARanywK6Hw4W2', 
  model="gpt-3.5-turbo"
)
print(job_created)
# job id='ftjob-eUZuStkelUGRhgeuSgv5xWYU'
# model id = ft:gpt-3.5-turbo-0125:rutgers-university::9SzpHo7I

##### partial sample5: done  #####
partial5_file = client.files.create(
  file=open("data/gptdata/partial5.jsonl", "rb"),
  purpose="fine-tune"
)
# FileObject(id='file-OcnvT7ZmRdKXU8M7cJ1tRYac', bytes=55092171, created_at=1716688867, filename='partial5.jsonl', object='file', purpose='fine-tune', status='processed', status_details=None)
job_created = client.fine_tuning.jobs.create(
  training_file='file-OcnvT7ZmRdKXU8M7cJ1tRYac', 
  model="gpt-3.5-turbo"
)
print(job_created)
# job id='ftjob-DUu4Mdi2gELL5nYGrcbTrXP7'
# model id = ft:gpt-3.5-turbo-0125:rutgers-university::9T0B0SCA

# client.fine_tuning.jobs.list(limit=10).data[0]
# client.fine_tuning.jobs.cancel("ftjob-abc123")


##### culture #####
contextualized_file = client.files.create(
  file=open("data/gptdata/culture.jsonl", "rb"),
  purpose="fine-tune"
)
# FileObject(id='file-fnuayfHRJunlPa2mteM91MxD', bytes=4694712, created_at=1716945080, filename='culture.jsonl', object='file', purpose='fine-tune', status='processed', status_details=None)
job_created = client.fine_tuning.jobs.create(
  training_file='file-fnuayfHRJunlPa2mteM91MxD', 
  model="gpt-3.5-turbo"
)
print(job_created)
# jobid = ftjob-sy8GwE46DIs4RE14TOiTI9dl
# model: 'ft:gpt-3.5-turbo-0125:rutgers-university::9U39iPwd'

##### math #####
contextualized_file = client.files.create(
  file=open("data/gptdata/math.jsonl", "rb"),
  purpose="fine-tune"
)
# FileObject(id='file-7B8ZfDWjn9dflKDPaJZX8pe6', bytes=4968642, created_at=1716945126, filename='math.jsonl', object='file', purpose='fine-tune', status='processed', status_details=None)
job_created = client.fine_tuning.jobs.create(
  training_file='file-7B8ZfDWjn9dflKDPaJZX8pe6', 
  model="gpt-3.5-turbo"
)
print(job_created)
# job id: ftjob-Es0tziWMyXrW3vziYqIQs4Gd
# model: 'ft:gpt-3.5-turbo-0125:rutgers-university::9U3NgfeC'

##### health #####
contextualized_file = client.files.create(
  file=open("data/gptdata/health.jsonl", "rb"),
  purpose="fine-tune"
)
# FileObject(id='file-zPQkTLrzBNAqb54M9opO1gtV', bytes=4743798, created_at=1716945755, filename='health.jsonl', object='file', purpose='fine-tune', status='processed', status_details=None)
job_created = client.fine_tuning.jobs.create(
  training_file='file-zPQkTLrzBNAqb54M9opO1gtV', 
  model="gpt-3.5-turbo"
)
# job id: ftjob-BorI6wpnhO69q4v9a9axxTkO
# model: 'ft:gpt-3.5-turbo-0125:rutgers-university::9U3Juq9u

##### technology #####
contextualized_file = client.files.create(
  file=open("data/gptdata/technology.jsonl", "rb"),
  purpose="fine-tune"
)
# FileObject(id='file-r7kZXGu3I6rgm27WR5uxernc', bytes=4873112, created_at=1717060022, filename='technology.jsonl', object='file', purpose='fine-tune', status='processed', status_details=None)
job_created = client.fine_tuning.jobs.create(
  training_file='file-r7kZXGu3I6rgm27WR5uxernc', 
  model="gpt-3.5-turbo"
)
# job id: ftjob-BorI6wpnhO69q4v9a9axxTkO
# model: 'ft:gpt-3.5-turbo-0125:rutgers-university::9U3Juq9u



##### culture + math #####
contextualized_file = client.files.create(
  file=open("data/gptdata/culture_math.jsonl", "rb"),
  purpose="fine-tune"
)
# FileObject(id='file-8eJmjprQ2Q3aMbW9lWBv9reI', bytes=4943970, created_at=1716946688, filename='culture_math.jsonl', object='file', purpose='fine-tune', status='processed', status_details=None)
job_created = client.fine_tuning.jobs.create(
  training_file='file-8eJmjprQ2Q3aMbW9lWBv9reI', 
  model="gpt-3.5-turbo"
)
# job id: ftjob-Izeci5utPYp2zvpA87OZesFe
# model id: ft:gpt-3.5-turbo-0125:rutgers-university::9U4hNB6I


# client.fine_tuning.jobs.list(limit=10).data[0]


##### level one + two #####
contextualized_file = client.files.create(
  file=open("data/gptdata/onetwo.jsonl", "rb"),
  purpose="fine-tune"
)
# FileObject(id='file-vNDuml2doYb2LYgdozLr0Vqu', bytes=2431978, created_at=1716967428, filename='onetwo.jsonl', object='file', purpose='fine-tune', status='processed', status_details=None)
job_created = client.fine_tuning.jobs.create(
  training_file='file-vNDuml2doYb2LYgdozLr0Vqu', 
  model="gpt-3.5-turbo"
)
# job id: ftjob-MWWxBpuz2D0n2MeMrKxq6OdT
# model id: 'ft:gpt-3.5-turbo-0125:rutgers-university::9U8toWRM'


##### level one + two + three #####
contextualized_file = client.files.create(
  file=open("data/gptdata/onetwothree.jsonl", "rb"),
  purpose="fine-tune"
)
# FileObject(id='file-F9FwR26Gy2Ec03hjAds7ille', bytes=3245010, created_at=1716967481, filename='onetwothree.jsonl', object='file', purpose='fine-tune', status='processed', status_details=None)
job_created = client.fine_tuning.jobs.create(
  training_file='file-F9FwR26Gy2Ec03hjAds7ille', 
  model="gpt-3.5-turbo"
)
# job id: ftjob-wEHhXRs049PmAp9StLSNLeLs
# model id: 'ft:gpt-3.5-turbo-0125:rutgers-university::9U8sGWqH'




# client.fine_tuning.jobs.list(limit=10).data[0]