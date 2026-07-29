import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
courses = pd.read_csv("courses.csv")

# Convert skills into vectors
vectorizer = TfidfVectorizer()

course_vectors = vectorizer.fit_transform(courses["Skills"])

# Take user input
user_input = input("Enter your interests: ")

user_vector = vectorizer.transform([user_input])

# Calculate similarity
similarity = cosine_similarity(user_vector, course_vectors)

# Get top 5 recommendations
scores = similarity.flatten()

top_courses = scores.argsort()[::-1][:5]

print("\nRecommended Courses:\n")

for i in top_courses:
    print(f"{courses.iloc[i]['Course']}  ({scores[i]:.2f})")