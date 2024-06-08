PARSE_PROMPTS = {
    "folio":
"""
Given a question-result pair, your task is to extract a concise answer from the given result.
If the result contains a concrete answer, the answer should be either true, false, or uncertain.
Return the answer using the following format: <answer>true</answer> or <answer>false</answer> or <answer>uncertain</answer>.
If the result does not provide a clear answer or the result is completely non-sense, then return <no_answer>.

Question:
{}

Result:
{}

Assistant:
""",

    "ruletaker":
"""
Given a question-result pair, your task is to extract a concise answer from the given result.
If the result contains a concrete answer, the answer should be either entailment or not entailment.
Return the answer using the following format: <answer>entailment</answer> or <answer>not entailment</answer>.
If the result does not provide a clear answer or the result is completely non-sense, then return <no_answer>.

Question:
{}

Result:
{}

Assistant:
""",

    "logicnli":
"""
Given a question-result pair, your task is to extract a concise answer from the given result.
If the result contains a concrete answer, the answer should be either contradiction, self-contradiction, neutral, or entailment.
Return the answer using the following format: <answer>contradiction</answer> or <answer>self-contradiction</answer> or <answer>neutral</answer> or <answer>entailment</answer>.
If the result does not provide a clear answer or the result is completely non-sense, then return <no_answer>.

Question:
{}

Result:
{}

Assistant:
""",

    "sst2":
"""
Given a question-result pair, your task is to extract a concise answer from the given result.
If the result contains a concrete answer, the answer should be either positive or negative.
Return the answer using the following format: <answer>positive</answer> or <answer>negative</answer>.
If the result does not provide a clear answer or the result is completely non-sense, then return <no_answer>.

Question:
{}

Result:
{}

Assistant:
""",

    "cola":
"""
Given a question-result pair, your task is to extract a concise answer from the given result.
If the result contains a concrete answer, the answer should be either Acceptable or Unacceptable.
Return the answer using the following format: <answer>Acceptable</answer> or <answer>Unacceptable</answer>.

Question:
{}

Result:
{}

Assistant:
""",

    "qqp":
"""
Given a question-result pair, your task is to extract a concise answer from the given result.
If the result contains a concrete answer, the answer should be either equivalent or not equivalent.
Return the answer using the following format: <answer>equivalent</answer> or <answer>not equivalent</answer>.

Question:
{}

Result:
{}

Assistant:
""",

    "wnli":
"""
Given a question-result pair, your task is to extract a concise answer from the given result.
If the result contains a concrete answer, the answer should be either entailment or not entailment.
Return the answer using the following format: <answer>entailment</answer> or <answer>not entailment</answer>.

Question:
{}

Result:
{}

Assistant:
""",

    "deductive_logic":
"""
Given a question-result pair, your task is to extract a concise answer from the given result.
If the result contains a concrete answer, the answer should be either True, False, or N/A.
Return the answer using the following format: <answer>True</answer> or <answer>False</answer> or <answer>N/A</answer>.
If the result does not provide a clear answer or the result is completely non-sense, then return <no_answer>.

Question: 
{}

Result:
{}

Assistant:
""",

    "abductive_logic":
"""
Given a question-result pair, your task is to extract a concise answer from the given result.
If the result contains a concrete answer, the answer should be either True, False, or N/A.
Return the answer using the following format: <answer>True</answer> or <answer>False</answer> or <answer>N/A</answer>.
If the result does not actually provide any answer or the result is completely non-sense, then return <no_answer>.

question: 
{}

result:
{}

Assistant:
""",

}

EVALUATE_PROMPTS = {
    "deductive_logic":
"""
Analyze and answer the deductive logical problem below step by step.

Deductive logic problem:
{}

Answer the question use: <answer>True</answer> or <answer>False</answer> or <answer>N/A</answer>, N/A means the result can not be deduced.
""",

    "abductive_logic":
"""
Analyze and answer the abductive logic problem below step by step.

Abductive logic problem:
{}

Answer the question use: <answer>True</answer> or <answer>False</answer> or <answer>N/A</answer>, N/A means the result can not be abduced.
""",
}

RAW_DATA_PROMPTS = {
    "arithmetic": "{descriptions}\nCompute the result of {vars}.",
    "deductive_logic": "{descriptions}\nDeduce the result of {vars}.",
    "abductive_logic": "{descriptions}.",
}


CONTEXTUALIZE_PROMPTS = {
    "deductive_logic": 
"""
Your task is to translate the following abstract deductive logic problem into a coherent and sensible contextualized logic problem in the domain of {}, which belongs to {} category. The brief explanation of the domain is: {}.
Do not create contextualized logic problems that are too generic or about a given definition; instead, create a scenario that is specific (about specific people or entities) and sensible.
The symbol '->' represents a deductive relationship, e.g., A -> B implies that if A is true, then B is true. If A is false, B's truth value remains undetermined (N/A). If the solution can not be deduced, answer 'N/A'.

Below is an example:

<example>

Human: 
Below is an abstract deductive logic reasoning problem:
aab is True.
aaa is False.
(aaa or aab) -> aac.
aad is True.
aae is True.
(aad and aae) -> aaf.
(aac or aaf) -> aag
What is the truth value of aag? (True or False or N/A (undetermined)).

Assistant:

The abstract variables in the abstract deductive logic can be instantiated as below:
<aab>John went to the store</aab>
<aaa>John bought milk</aaa>
<aac>John bought something</aac>
<aad>John bought bread</aad>
<aae>John paid with cash</aae>
<aaf>John made a purchase</aaf>
<aag>John spent money</aag>

Translating into coherent natural language:
<nl>John went to the store but he did not buy milk. 
If John bought milk or went to the store, then it means he bought something. 
John also bought bread, paid with cash. If John bought bread and paid with cash, then it means he made a purchase. 
If John bought something or he made a purchase, then he spent some money.
Does John spend any money?</nl>

</example>

Here is a new abstract deductive logic problem:

{}

Translate the abstract deductive logic into a contextualized deductive logic problem in the domain of {}, which belongs to {} category. Please follow the above format with xml tags.
""",

    "abductive_logic":
"""
Your task is to translate the following abstract abductive logic problem into a coherent and sensible contextualized logic problem in the domain of {}, which belongs to {} category. The brief explanation of the domain is: {}.
The symbol '->' represents a deductive relationship, e.g., A -> B implies that if B is false, then A is false. If B is true, A's truth value remains undetermined (N/A). If the solution can not be abduced, answer 'N/A'.

Below is an example:

<example>

Human: 
Below is an abstract abductive logic reasoning problem:
(NOT aah) -> aai.
(aaa or aab) -> aac.
(NOT aac) -> aad.
(aae or aaf) -> aag.
(aag or aai) -> aaj.
(aad or aaj) -> aak.
Given aak is False, what is the value of aah? (True or False or N/A (undetermined)).

Assistant:

Let's translate the abstract abductive logic problem into a coherent and natural language scenario.

<NOT aah>The library is not open late.</NOT aah>
<aai>Laura does not have a place to study.</aai>
<aaa>Laura has a math exam tomorrow.</aaa>
<aab>Laura has a history essay due tomorrow.</aab>
<aac>Laura needs to study tonight.</aac>
<NOT aac>Laura decides to watch a movie instead.</NOT aac>
<aad>Laura is not preparing for her exams.</aad>
<aae>Laura has group study plans for history.</aae>
<aaf>Laura's friend has offered to help with math.</aaf>
<aag>Laura has study support.</aag>
<aag or aai>Laura has a reason to stay home.</aag or aai>
<aad or aaj>Laura is not focusing on her academics.</aad or aaj>
<aak>Laura is completely neglecting her studies.</aak>


Translating into coherent natural language:
<nl>
The library is not open late, which means Laura does not have a place to study. Laura has both a math exam and a history essay due tomorrow, which implies she needs to study tonight. However, if Laura does not need to study tonight, then she decides to watch a movie instead, indicating she is not preparing for her exams.
Laura has group study plans for history and her friend has offered to help with math, meaning she has study support. If Laura has study support or does not have a place to study, then she has a reason to stay home. If Laura is not preparing for her exams or she has a reason to stay home, then she is not focusing on her academics.
Given that Laura is not completely neglecting her studies is false, what can be said about the library being open late? (True, False, or N/A (undetermined)).
</nl>


</example>

Here is a new abstract abductive logic problem:

{}

Translate the abstract abductive logic into a contextualized deductive logic problem in the domain of {}, which belongs to {} category. Please follow the above format with xml tags.
""",

}

# COMBINE_VARIABLE_PROMPTS = {
#     "deductive_logic": 
# """
# Your task is to translate each one of the abstract variables in the following abstract deductive logic problem into coherent and sensible contextualized sentences in the domain of {}, which belongs to {} category. The brief explanation of the domain is: {}.
# Do not create contextualized sentences that are too generic or about a given definition; instead, create a scenario that is specific (about specific people or entities) and sensible.
# The symbol '->' represents a deductive relationship, e.g., A -> B implies that if A is true, then B is true. If A is false, B's truth value remains undetermined (N/A). If the solution can not be deduced, answer 'N/A'.

# Below is an example:

# <example>

# Human: 
# Below is an abstract deductive logic reasoning problem:
# aab is True.
# aaa is False.
# (aaa or aab) -> aac.
# aad is True.
# aae is True.
# (aad and aae) -> aaf.
# (aac or aaf) -> aag
# What is the truth value of aag? (True or False or N/A (undetermined)).

# Assistant:

# The variables contained in the above deductive logic are: aab, aaa, aac, aad, aae, aaf, aag.

# The abstract variables in the abstract deductive logic can be instantiated as below:
# <aab>John went to the store</aab>
# <aaa>John bought milk</aaa>
# <aac>John bought something</aac>
# <aad>John bought bread</aad>
# <aae>John paid with cash</aae>
# <aaf>John made a purchase</aaf>
# <aag>John spent money</aag>

# </example>

# Here is a new abstract deductive logic problem:

# {}

# Translate each one of the abstract variables in the following abstract deductive logic problemthe abstract deductive logic into a contextualized deductive logic problem in the domain of {}, which belongs to {} category. Please follow the above format with xml tags.
# """,

#     "abductive_logic":
# """
# Your task is to translate the following abstract abductive logic problem into a coherent and sensible contextualized logic problem in the domain of {}, which belongs to {} category. The brief explanation of the domain is: {}.
# The symbol '->' represents a deductive relationship, e.g., A -> B implies that if B is false, then A is false. If B is true, A's truth value remains undetermined (N/A). If the solution can not be abduced, answer 'N/A'.

# Below is an example:

# <example>

# Human: 
# Below is an abstract abductive logic reasoning problem:
# (NOT aah) -> aai.
# (aaa or aab) -> aac.
# (NOT aac) -> aad.
# (aae or aaf) -> aag.
# (aag or aai) -> aaj.
# (aad or aaj) -> aak.
# Given aak is False, what is the value of aah? (True or False or N/A (undetermined)).

# Assistant:

# Let's translate the abstract abductive logic problem into a coherent and natural language scenario.

# <NOT aah>The library is not open late.</NOT aah>
# <aai>Laura does not have a place to study.</aai>
# <aaa>Laura has a math exam tomorrow.</aaa>
# <aab>Laura has a history essay due tomorrow.</aab>
# <aac>Laura needs to study tonight.</aac>
# <NOT aac>Laura decides to watch a movie instead.</NOT aac>
# <aad>Laura is not preparing for her exams.</aad>
# <aae>Laura has group study plans for history.</aae>
# <aaf>Laura's friend has offered to help with math.</aaf>
# <aag>Laura has study support.</aag>
# <aag or aai>Laura has a reason to stay home.</aag or aai>
# <aad or aaj>Laura is not focusing on her academics.</aad or aaj>
# <aak>Laura is completely neglecting her studies.</aak>


# Translating into coherent natural language:
# <nl>
# The library is not open late, which means Laura does not have a place to study. Laura has both a math exam and a history essay due tomorrow, which implies she needs to study tonight. However, if Laura does not need to study tonight, then she decides to watch a movie instead, indicating she is not preparing for her exams.
# Laura has group study plans for history and her friend has offered to help with math, meaning she has study support. If Laura has study support or does not have a place to study, then she has a reason to stay home. If Laura is not preparing for her exams or she has a reason to stay home, then she is not focusing on her academics.
# Given that Laura is not completely neglecting her studies is false, what can be said about the library being open late? (True, False, or N/A (undetermined)).
# </nl>


# </example>

# Here is a new abstract abductive logic problem:

# {}

# Translate the abstract abductive logic into a contextualized deductive logic problem in the domain of {}, which belongs to {} category. Please follow the above format with xml tags.
# """,

# }


LOGIC_FOLLOWING_CHECKING = """
Your task is to check whether the contextualized logic problem follows exactly the same logical structure as the abstract logic problem.
You should answer <answer>Yes</answer> if the contextualized logic problem follows the same logical structure as the abstract logic problem, and <answer>No</answer> otherwise.

Below are two examples:

<example>
Human:

The abstract logic problem is below:

aaa is True.
aab is False.
(aaa and aab) -> aac.
aad is True.
aae is True.
(aad or aae) -> aaf.
(aac and aaf) -> aag.
Is the truth value of aag True/False/(N/A)?

The contextualized logic problem is below:

Mary studied hard for the test, but the test turned out not to be easy.
If Mary studied hard and the test was easy, then she would have gotten an A on the test.
Mary is smart, and she always does her homework. If Mary is smart or she always does her homework, then she is a good student.
If Mary got an A on the test and she is a good student, then her parents will be proud of her.
Will Mary's parents be proud of her?

Assistant:
aaa corresponds to "Mary studied hard for the test".
aab corresponds to "the test being easy".
aab is False corresponds to "the test turned out not to be easy".
aac corresponds to "Mary getting an A on the test".
(aaa and aab) -> aac corresponds to "If Mary studied hard and the test was easy, then she would have gotten an A on the test.".
aad corresponds to "Mary being smart".
aae corresponds to "Mary always doing her homework".
aaf corresponds to "Mary being a good student".
(aad or aae) -> aaf corresponds to "If Mary is smart or she always does her homework, then she is a good student".
(aac and aaf) -> aag corresponds to "If Mary got an A on the test and she is a good student, then her parents will be proud of her".
aag corresponds to "Mary's parents being proud of her".
the question about the truth value of aag is equivalent to the question about whether Mary's parents will be proud of her.

The answer is <answer>Yes</answer>.
</example>

<example>
Human:

The abstract logic problem is below:

aah is False.
aak is False.
aan is True.
(not aak or not aan) -> not aao.
aai is True.
(aah or aai) -> aaj.
(aaj or aao) -> aap.
aaa is False.
aab is True.
(aaa and aab) -> aac.
aae is False.
aad is False.
(aad or aae) -> aaf.
(aac and aaf) -> aag.
(aag or aap) -> aaq.
Is the truth value of aaq True/False/(N/A)?

The contextualized logic problem is below:

Mary did not study for the math test. Mary also did not get enough sleep the night before, although she did eat a good breakfast the morning of the test. If Mary got enough sleep and ate a good breakfast, then she would have been well-prepared for the test. 
Mary arrived to the math test on time. If Mary either studied or arrived on time, then she was present for the test. If Mary was present for the test or was well-prepared, then she took the test.
Mary did not find the math test to be easy, but she did finish within the time limit. If the test was easy for Mary and she finished in time, then she would have completed the test. 
Mary did not double-check her answers or catch all her mistakes. If she had caught her mistakes or double-checked, she would have been careful on the test. If Mary completed the test carefully, she would have done well.
Finally, if Mary either did well on the test or at least took the test, then she would pass.
Given the information above, did Mary pass the math test?

Assistant:
aah corresponds to "Mary studying for the math test".
aah is False corresponds to "Mary did not study for the math test".
aak corresponds to "Mary getting enough sleep".
aak is False corresponds to "Mary did not get enough sleep".
aan corresponds to "Mary eating a good breakfast".
aao corresponds to "Mary being well-prepared for the test".
(not aak or not aan) -> not aao should correspond to "If Mary did not get enough sleep or did not eat a good breakfast, then she would not have been well-prepared for the test".
However, the contextualized logic problem does not include this condition, but instead includes a different condition: "If Mary got enough sleep and ate a good breakfast, then she would have been well-prepared for the test".
Thus the contextualized logic problem does not follow the same logical structure as the abstract logic problem.

The answer is <answer>No</answer>.

</example>


Now here is a new abstract logic problem and a new contextualized logic problem. Please check whether the contextualized logic problem follows the same logical structure as the abstract logic problem.

The abstract logic problem is below:

{}

The contextualized logic problem is below:

{}

Assistant:
"""


COMMON_SENSE_CHECKING = """
Given a logic problem asking about whether a statement is true or false, your task is to check whether the question being asked is actually about a known knowledge.
You should answer <answer>Yes</answer> if the problem is asking about a known knowledge, and <answer>No</answer> otherwise.

Below are two examples:

<example>
Human:

Here is a logic problem:

The Moon does orbit the Earth.
If the Moon does not orbit the Earth, then it would mean the Moon is not gravitationally bound to Earth.
Is it true that the Moon is not gravitationally bound to Earth?

Assistant:

The question being asked in the logic problem is "Is it true that the Moon is not gravitationally bound to Earth?".
As we all know that the Moon orbits the Earth, it is common sense that the Moon is gravitationally bound to Earth.
Thus the problem is indeed asking about a known knowledge.
So the answer is <answer>Yes</answer>.

</example>

<example>

Human:

Here is a logic problem:

The stock market is up today.
If the stock market is not up today, then John and Andy will be selling their stocks. 
Given the stock market is in fact up today, are John and Andy selling their stocks?

Assistant:

The question being asked in the logic problem is "Are John and Andy selling their stocks?".
The problem is not asking about a known knowledge, but rather about a hypothetical situation and about unknown people John and Andy.
Thus the problem is not asking about a known knowledge.
So the answer is <answer>No</answer>.

</example>

Now here is a new logic problem. Please check whether the question being asked is about a known knowledge.

Here is a logic problem:

{}

Assistant:
"""


SENSIBILITY_CHECKING = """
Given a logic problem, your task is to check whether the contextualized logic problem is sensible and coherent.
You should answer <answer>Yes</answer> if the contextualized logic problem is sensible and coherent, and <answer>No</answer> otherwise.

Below are two examples:

<example>
Human:

Here is a logic problem:

Amy felt too much anxious during her presentation.
If Amy did not feel anxious during her presentation, then her anxiety improved her presentation performance.
Did Amy's anxiety improve her presentation performance?

Assistant:

This problem does not really make sense because of this condition: "If Amy did not feel anxious during her presentation, then her anxiety improved her presentation performance."
It does not make sense because if Amy did not feel anxious, then how could her anxiety improve her presentation performance?
Thus the contextualized logic problem is not sensible.
So the answer is <answer>No</answer>.
</example>

<example>
Human:

Here is a logic problem:

John was at the crime scene when the crime occurred.
If John was not at the crime scene, then John must have an alibi.
Does John have an alibi?

Assistant:

Yes, this problem is sensible and coherent.
So the answer is <answer>Yes</answer>.

</example>

Now here is a new logic problem. Please check whether the contextualized logic problem is sensible and coherent.

Here is a logic problem:

{}

Assistant:
"""


TAUTOLOGY_CHECKING = """
Given a logic problem, your task is to check whether the contextualized logic problem contains if-conditions that are actually tautologies.
You should answer <answer>Yes</answer> if the contextualized logic problem does contain a tautology, and <answer>No</answer> otherwise.

Below are two examples:

<example>
Human:

Here is a logic problem:

HFCl is a strong acid that fully dissociates in water.\nIf HFCl acid is not a strong acid, then it must be a weak acid.\nIs HFCl acid a weak acid?

Assistant:
This problem contains the condition "If HFCl acid is not a strong acid, then it must be a weak acid".
This condition is a tautology because if HFCl acid is not a strong acid, then it must be a weak acid.
Thus it corresponds to a logical form of "not p -> not p" letting p be the statement "HFCl acid is a strong acid".
So the answer is <answer>Yes</answer>.
</example>

<example>
Human:

Here is a logic problem:

The Sun rises in the east.\nIf the Sun does not rise in the east, then it must rise in the west.\nDoes the Sun rise in the west?

Assistant:
This problem does not contain a tautology.
So the answer is <answer>No</answer>.
</example>

Now here is a new logic problem. Please check whether the contextualized logic problem contains a tautology.

{}

Assistant:
"""


A1_EXAMPLES = """
<example>
Human: 
Here are the sentence and the vocabulary from A1 level:

sentence: Alice's bank account balance is zero
vocabulary: ['house', 'cloud', 'big', 'side', 'put', 'keep', 'store', 'art', 'often', 'that', 'important', 'sum', 'nice', 'middle', 'circle', 'something', 'egg', 'out', 'in', 'game', 'nothing', 'word', 'number', 'where', 'two', 'result', 'set', 'large', 'one', 'list', 'busy', 'hill', 'ground', 'spot', 'make', 'much', 'all', 'up', 'turn', 'full', 'of', 'city', 'than', 'history', 'new', 'people', 'work', 'no', 'before', 'with', 'thing', 'inside', 'at', 'judge', 'for', 'town', 'ring', 'story', 'wheel', 'review', 'many', 'building', 'mine', 'has', 'money', 'piece', 'bank', 'information', 'order', 'lot', 'to', 'skill', 'item', 'answer', 'reason', 'love', 'business', 'place', 'site', 'table', 'office', 'save', 'you', 'river', 'everyone', 'card', 'file', 'line', 'there', 'paper', 'first', 'group', 'shop', 'home', 'or', 'sight', 'difference', 'computer', 'person']

Assistant:
rewritten_sentence: 
<r>Alice has no money in the bank.</r>
</example>

<example>
Human: 
Here are the sentence and the vocabulary from A1 level:

sentence: Mary completed her homework assignment
vocabulary: ['busy', 'teacher', 'homework', 'home', 'job', 'exercise', 'right', 'work', 'school']

Assistant:
rewritten_sentence: 
<r>Mary did her homework.</r>
</example>

<example>
Human: 
Here are the sentence and the vocabulary from A1 level:

sentence: The defendant had a motive for the crime
vocabulary: ['wrong', 'of', 'party', 'subject', 'cookie', 'person', 'and', 'bad', 'thing', 'idea', 'move', 'in', 'habit', 'reason', 'social', 'practice', 'activity', 'problem']

Assistant:
rewritten_sentence: 
<r>The defendant had a reason for the crime.</r>
</example>

<example>
Human: 
Here are the sentence and the vocabulary from A1 level:

sentence: The defendant is innocent
vocabulary: ['wrong', 'of', 'party', 'child', 'in', 'young', 'not', 'person']

Assistant:
rewritten_sentence: 
<r>The defendant has nothing wrong.</r>
</example>

<example>
Human: 
Here are the sentence and the vocabulary from A1 level:

sentence: The defendant has an alibi
vocabulary: ['wrong', 'party', 'excuse', 'in', 'person']

Assistant:
This sentence cannot be rewritten using the given vocabulary without changing the meaning or becoming a strange sentence.
<None>
</example>
"""


A2_EXAMPLES = """
<example>
Human: 
Here are the sentence and the vocabulary from A2 level:

sentence: Sarah brings an umbrella
vocabulary: ['cover', 'provide', 'shade', 'range']

Assistant:
This sentence cannot be rewritten using the given vocabulary without changing the meaning or being very strange.
<None>
</example>

<example>
Human: 
Here are the sentence and the vocabulary from A2 level:

sentence: The defendant had a motive for the crime
vocabulary: ['specific', 'move', 'complaint', 'hate', 'crime', 'being', 'in', 'communication', 'entertainment', 'law', 'web', 'murder', 'fool', 'cause', 'robbery', 'act', 'wrong', 'inspiration', 'illegal', 'film', 'situation']

Assistant:
rewritten_sentence: 
<r>The defendant had a cause for the crime.</r>
</example>

<example>
Human: 
Here are the sentence and the vocabulary from A2 level:

sentence: The defendant is innocent
vocabulary: ['complaint', 'being', 'in', 'opposite', 'wrong']

Assistant:
This sentence cannot be rewritten using the given vocabulary without changing the meaning or being very strange.
<None>
</example>

<example>
Human: 
Here are the sentence and the vocabulary from A2 level:

sentence: The defendant was at the crime scene
vocabulary: ['specific', 'fiction', 'complaint', 'area', 'hate', 'chapter', 'crime', 'pretty', 'being', 'in', 'agent', 'entertainment', 'law', 'part', 'web', 'murder', 'slide', 'fool', 'view', 'scenery', 'landscape', 'robbery', 'act', 'wrong', 'fit', 'background', 'illegal', 'balcony', 'film', 'one']

Assistant:
rewritten_sentence: 
<r>The defendant was at the area of the crime</r>
</example>

<example>
Human: 
Here are the sentence and the vocabulary from A2 level:

sentence: The defendant has an alibi
vocabulary: ['complaint', 'being', 'in', 'film', 'wrong']

Assistant:
This sentence is difficult to be rewritten using this words.
<None>
</example>
"""


B1_EXAMPLES = """
<example>
Human: 
Here are the sentence and the vocabulary from B1 level:

sentence: Alice studied for the exam
vocabulary: ['examination', 'conscious']

Assistant:
<r>Alice studied for the examination</r>
</example>

<example>
Human: 
Here are the sentence and the vocabulary from B1 level:

sentence: Alice will fail the exam
vocabulary: ['fall', 'financial', 'examination', 'failure', 'quality', 'disappointment', 'disappoint', 'loss', 'result', 'academic', 'flunk']

Assistant:
rewritten_sentence: 
<r>Alice will flunk the examination</r>
</example>

<example>
Human: 
Here are the sentence and the vocabulary from B1 level:

sentence: The defendant is innocent
vocabulary: ['criminal', 'guilty', 'pure', 'opposite', 'guilt', 'responsibility', 'satisfaction', 'civil']

Assistant:
rewritten_sentence:
<r>The criminal is not guilty.</r>
</example>

<example>
Human: 
Here are the sentence and the vocabulary from B1 level:

sentence: The defendant has an alibi
vocabulary: ['criminal', 'excuse', 'satisfaction', 'film', 'civil']

Assistant:
rewritten_sentence:
<r>The defendant has an excuse</r> 
</example>

<example>
Human: 
Here are the sentence and the vocabulary from B1 level:

sentence: The defendant should be acquitted
vocabulary: ['satisfaction', 'criminal', 'civil', 'guilty']

Assistant:
rewritten_sentence:
<r>The criminal should not be found guilty.</r>
</example>
"""



B2_EXAMPLES = """
<example>
Human: 
Here are the sentence and the vocabulary from B2 level:

sentence: Alice's bank account balance is zero
vocabulary: ['institution', 'balance', 'compensate', 'make', 'reckon', 'rehearsal', 'cast', 'deposit', 'trust', 'nought', 'line']

Assistant:
rewritten_sentence: 
<r>Alice's bank account balance is nought</r>
</example>

<example>
Human: 
Here are the sentence and the vocabulary from B2 level:

sentence: John lost money in the stock market today
vocabulary: ['stock', 'store', 'certificate', 'breed', 'standard']

Assistant:
rewritten_sentence: 
<r>John's stock certificates lost value on the standard market today.</r>
</example>

<example>
Human: 
Here are the sentence and the vocabulary from B2 level:

sentence: The patient likely has COVID-19
vocabulary: ['potential', 'probable']

Assistant:
rewritten_sentence: 
<r>The patient probably has COVID-19.</r>
</example>

<example>
Human: 
Here are the sentence and the vocabulary from B2 level:

sentence: Alice is experiencing depression
vocabulary: ['disorder']

Assistant:
rewritten_sentence: 
<r>Alice is experiencing a depressive disorder.</r>
</example>

<example>
Human: 
Here are the sentence and the vocabulary from B2 level:

sentence: Hydrochloric acid is a strong acid
vocabulary: ['potent', 'acid', 'forcefully']

Assistant:
rewritten_sentence: 
<r>Hydrochloric acid is a potent acid.</r>
</example>
"""

C1_EXAMPLES = """
<example>
Human: 
Here are the sentence and the vocabulary from C1 level:

sentence: The house is located in a good school district.
vocabulary: ['firm', 'house']

Assistant:
rewritten_sentence: 
<r>The house is situated in a school district of firm repute.</r>
</example>

<example>
Human: 
Here are the sentence and the vocabulary from C1 level:

sentence: The stock market is up today.
vocabulary: ['sprout', 'stock']

Assistant:
rewritten_sentence: 
<r>The stock market has sprouted upwards today.</r>
</example>

<example>
Human: 
Here are the sentence and the vocabulary from C1 level:

sentence: A star is in the main sequence stage of its life cycle.
vocabulary: ['succession']

Assistant:
This sentence cannot be rewritten using the given vocabulary without changing the meaning.
<None>
</example>

<example>
Human: 
Here are the sentence and the vocabulary from C1 level:

sentence: Hydrochloric acid is a strong acid.
vocabulary: ['durable', 'acidic']

Assistant:
rewritten_sentence: 
<r>Hydrochloric acid is a durable, strongly acidic compound.</r>
</example>

<example>
Human: 
Here are the sentence and the vocabulary from C1 level:

sentence: Sodium hydroxide is a strong base.
vocabulary: ['durable']

Assistant:
This sentence cannot be rewritten using the given vocabulary without changing the meaning.
<None>
</example>
"""



C2_EXAMPLES = """
<example>
Human: 
Here are the sentence and the vocabulary from C2 level:

sentence: The defendant is innocent
vocabulary: ['ingenuous']

Assistant:
rewritten_sentence: 
<r>The defendant is ingenuous.</r>
</example>

<example>
Human: 
Here are the sentence and the vocabulary from C2 level:

sentence: Alice's bank account balance is zero.
vocabulary: ['poise', 'equilibrium', 'additive']

Assistant:
This sentence cannot be rewritten using the given vocabulary without changing the meaning.
<None>
</example>

<example>
Human: 
Here are the sentence and the vocabulary from C2 level:

sentence: The patient has an infection.
vocabulary: ['contagion']

Assistant:
rewritten_sentence: 
<r>The patient has a contagion.</r>
</example>

<example>
Human: 
Here are the sentence and the vocabulary from C2 level:

sentence: Alice is experiencing depression
vocabulary: ['depressive']

Assistant:
rewritten_sentence: 
<r>Alice is grappling with a depressive episode.</r>
</example>

<example>
Human: 
Here are the sentence and the vocabulary from C2 level:

sentence: The house is located in a good school district
vocabulary: ['shoal', 'commodity']

Assistant:
This sentence cannot be rewritten using the given vocabulary without changing the meaning.
<None>
</example>
"""


RECOMBINE_PROMPT = """
Combine the following sentences into a coherent and sensible logic problem based on the abstract logic template.
Surround the combined result with <nl> and </nl> tags.

Below are some examples, where the sentences are combined to a coherent and sensible logic problem based on the abstract logic template:

<example>
Human: 

Sentences:
<aaa>Mary completed her homework assignment</aaa>
<aab>Mary will flunk the assignment</aab>

Abstract logic template:
aaa is True
(NOT aaa) -> aab
What is the truth value of aab?

Assistant:
We can first simply replace the variables in the abstract logic template with the corresponding sentences:

Mary completed her homework assignment (corresond to aaa is True.)
If not Mary completed her homework assignment, then Mary will flunk the assignment (corresponde to (NOT aaa) -> aab)
Will Mary flunk the assignment? (corresond to What is the truth value of aab?)

Then to make the above paragraph more coherent and natural, we can rewrite it as below:

<nl>Mary completed her homework assignment.
If Mary did not complete her homework assignment, then she will flunk the assignment.
Will Mary flunk the assignment?</nl>

</example>


<example>
Human: 

Sentences:
<aab>John ventured out to the marketplace to procure some essential goods.</aab>
<aaa>John acquired a container of milk.</aaa>
<aac>John did not leave the store empty-handed; he successfully acquired at least one item.</aac>
<aad>John has purchased a freshly baked loaf of bread.</aad>
<aae>John opted to pay for his purchases with paper currency.</aae>
<aaf>John's transaction was successful.</aaf>
<aag>John had expended some of his financial resources to obtain the goods he desired.</aag>

Abstract logic template:
aab is True.
aaa is False.
(aaa or aab) -> aac.
aad is True.
aae is True.
(aad and aae) -> aaf.
(aac or aaf) -> aag
What is the truth value of aag?

Assistant:
We can first simply replace the variables in the abstract logic template with the corresponding sentences:

John ventured out to the marketplace to procure some essential goods. (corresond to aab is True.)
John did not acquire a container of milk. (corresond to aaa is False.)
If John ventured out to the marketplace to procure some essential goods or John acquired a container of milk, then John did not leave the store empty-handed; he successfully acquired at least one item. (corresond to (aaa or aab) -> aac.)
John has purchased a freshly baked loaf of bread. (corresond to aad is True.)
John opted to pay for his purchases with paper currency. (corresond to aae is True.)
If John has purchased a freshly baked loaf of bread and John opted to pay for his purchases with paper currency, then John's transaction was successful. (corresond to (aad and aae) -> aaf.)
If John did not leave the store empty-handed; he successfully acquired at least one item or John's transaction was successful, then John had expended some of his financial resources to obtain the goods he desired. (corresond to (aac or aaf) -> aag)
Had John expended some of his financial resources to obtain the goods he desired? (corresond to What is the truth value of aag?)

Then to make the above paragraph more coherent and natural, we can rewrite it as below:

<nl>John went to the store but he did not acquire a container of milk.
If John went to the store or he acquired a container of milk, then he did not leave the store empty-handed.
John has purchased a freshly baked loaf of bread, opted to pay for his purchases with paper currency. 
If John has purchased a freshly baked loaf of bread and he opted to pay for his purchases with paper currency, then his transaction was successful.
If John did not leave the store empty-handed or his transaction was successful, then he had expended some of his financial resources to obtain the goods he desired.
Had John expended some of his financial resources to obtain the goods he desired?</nl>

</example>

<example>
Human: 

Sentences:
<aaa>It is raining outdoors</aaa>
<aab>Sarah brings a device for protection</aab>

Abstract logic template:
aaa is True
(NOT aaa) -> aab
What is the truth value of aab?

Assistant:
We can first simply replace the variables in the abstract logic template with the corresponding sentences:

It is raining outdoors (corresond to aaa is True.)
If not it is raning outdoors, then Sarah brings a device for protection (corresponde to (NOT aaa) -> aab)
Does Sarah bring a device for protection? (corresond to What is the truth value of aab?)

Then to make the above paragraph more coherent and natural, we can rewrite it as below:

<nl>It is raining outdoors.
If not it is raning outdoors, then Sarah brings a device for protection.
Does Sarah bring a device for protection?</nl>

</example>

Here are a new set of sentences and the abstract logic template. Combine the sentences into a coherent and sensible logic problem based on the abstract logic template following the above procedure and format.
Do not answer the problem, but only provide the coherent and sensible logic problem.
Surround the combined result with <nl> and </nl> tags following the above examples.

Sentences:
{}

Abstract logic template:
{}

Assistant:
"""