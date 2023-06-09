import spacy
nlp = spacy.load('en_core_web_md')

word1 = nlp("cat")
word2 = nlp("monkey")
word3 = nlp("banana")

print("---")
print(word1.similarity(word2))
print(word3.similarity(word2))
print(word3.similarity(word1))

tokens = nlp('cat apple monkey banana ')

print("---")
for token1 in tokens:
    for token2 in tokens:
        print(token1.text, token2.text, token1.similarity(token2))

sentence_to_compare = "Why is my cat on the car"

sentences = ["where did my dog go",
"Hello, there is my car",
"I\'ve lost my car in my car",
"I\'d like my boat back",
"I will name my dog Diana"]

model_sentence = nlp(sentence_to_compare)

print("---")
for sentence in sentences:
    similarity = nlp(sentence).similarity(model_sentence)
    print(sentence + " - ", similarity)

# The interesting thing about the similarities between "cat," "monkey," and "banana" is that 
# "cat" and "monkey" have a high similarity score of 0.800, meaning they have similar 
# semantic properties. On the other hand, "banana" has a lower similarity core of 0.671 with both 
# "cat" and "monkey." This means that "banana" shares fewer semantic similarities with the other two words.


# My own example
myword1 = nlp("car")
myword2 = nlp("bicycle")
myword3 = nlp("aeroplane")

print("---")
print(myword1.similarity(myword2))
print(myword3.similarity(myword2))
print(myword3.similarity(myword1))


# Comparing using SM instead of MD

nlp_sm = spacy.load('en_core_web_sm')

word1_sm = nlp_sm("cat")
word2_sm = nlp_sm("monkey")
word3_sm = nlp_sm("banana")

print("---")
print(word1_sm.similarity(word2_sm))
print(word3_sm.similarity(word2_sm))
print(word3_sm.similarity(word1_sm))

tokens_sm = nlp_sm('cat apple monkey banana ')

print("---")
for token1_sm in tokens_sm:
    for token2_sm in tokens_sm:
        print(token1_sm.text, token2_sm.text, token1_sm.similarity(token2_sm))

sentence_to_compare_sm = "Why is my cat on the car"

sentences_sm = [
    "where did my dog go",
    "Hello, there is my car",
    "I've lost my car in my car",
    "I'd like my boat back",
    "I will name my dog Diana"
]

model_sentence_sm = nlp_sm(sentence_to_compare_sm)

print("---")
for sentence_sm in sentences_sm:
    similarity_sm = nlp_sm(sentence_sm).similarity(model_sentence_sm)
    print(sentence_sm + " - ", similarity_sm)

# When running the code with 'en_core_web_sm', I observed that the similarity differed
# from those obtained with 'en_core_web_md'. Since 'en_core_web_sm' is a smaller model, it has a 
# limited representation of the semantic relationships between words, leading to potentially different 
# similarity scores. It struggles to provide accurate similarity scores for words that are not part of 
# its limited vocabulary. 