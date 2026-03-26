import json

prompts = {
    "prompts": [
        # --- Code & System (13 prompts) ---
        "Write a Python function to merge overlapping intervals and explain its time and space complexity.",
        "Implement a thread-safe singleton pattern in C++ and explain the memory barrier usage.",
        "Write a bash script that finds all .log files modified in the last 7 days and compresses them into a tar.gz.",
        "Explain the differences between epoll, kqueue, and select in Linux/Unix socket programming.",
        "Write a React hook `useDebounce` that takes a value and a delay, and returns the debounced value.",
        "Explain how continuous batching (iteration-level scheduling) works in LLM inference engines like vLLM.",
        "Write a Go script to concurrently fetch 10 URLs and aggregate the status codes using channels.",
        "Explain the CAP theorem and provide an example of a CP system versus an AP system.",
        "Write a Dockerfile for a Python FastAPI application using a multi-stage build.",
        "Explain the difference between a process and a thread in an operating system.",
        "Write a SQL query to find the second highest salary from an Employee table without using LIMIT.",
        "Describe how a bloom filter works and what its primary use cases are in distributed systems.",
        "Write a short Python script using the `csv` and `json` modules to parse a standard CSV file and convert it to a JSON array of objects.",
    
        # --- Math & Reasoning (12 prompts) ---
        "Solve this step by step: if a model emits 128 tokens in 2.5 seconds, what is the throughput in tokens per second?",
        "A train leaves New York at 8:00 AM traveling 60 mph. Another leaves at 9:00 AM traveling 80 mph. When will the second catch the first?",
        "If 3x + 4y = 10 and 2x - y = 5, what are the values of x and y? Show your mathematical work step by step.",
        "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?",
        "If you have a 5-liter jug and a 3-liter jug, and an unlimited supply of water, how do you measure exactly 4 liters?",
        "Explain the Monty Hall problem and mathematically prove why switching doors increases your odds of winning.",
        "Calculate the derivative of f(x) = x^3 * sin(x) using the product rule.",
        "In a room of 23 people, what is the approximate probability that at least two people share a birthday? Show the calculation.",
        "If a triangle has sides of length 3, 4, and 5, is it a right triangle? Prove it using the Pythagorean theorem.",
        "A bag contains 5 red marbles, 3 blue marbles, and 2 green marbles. What is the probability of drawing a blue marble?",
        "Solve for x in the equation 2^(x+1) = 16.",
        "If a widget factory produces 100 widgets in 5 hours, how many widgets does it produce in 12 hours?",
    
        # --- Science & Academic (12 prompts) ---
        "Compare photosynthesis and cellular respiration in a concise, structured explanation.",
        "Explain the concept of quantum entanglement and why Einstein referred to it as 'spooky action at a distance'.",
        "Describe the theoretical mechanism of action for mRNA vaccines, using COVID-19 as an example.",
        "Explain the differences between Special and General Relativity in theoretical physics.",
        "What is the Fermi Paradox, and what are three popular hypotheses that attempt to resolve it?",
        "Describe the structure of a transformer block in deep learning, including self-attention and feed-forward layers.",
        "Explain the process of plate tectonics and how it leads to the formation of oceanic trenches.",
        "What are the main differences between meiosis and mitosis?",
        "Explain the central dogma of molecular biology.",
        "What is the difference between nuclear fission and nuclear fusion?",
        "Describe the stages of the water cycle.",
        "Explain the greenhouse effect and its relation to climate change.",
    
        # --- Translation & Linguistics (11 prompts) ---
        "Translate the following English sentence into French, Spanish, and German: 'The migration was completed successfully without any downtime.'",
        "Explain the difference between a morpheme and a phoneme using examples.",
        "Translate this idiomatic expression to English and explain its origin: 'Coup de foudre'.",
        "What is the Sapir-Whorf hypothesis? Provide arguments for and against it.",
        "Translate this paragraph to Japanese: 'Speculative decoding uses a smaller draft model to predict the next few tokens, which are then verified by the target model in parallel.'",
        "Explain the grammatical cases in the German language (Nominative, Accusative, Dative, Genitive) with examples.",
        "Summarize the differences between logographic, syllabic, and alphabetic writing systems.",
        "Translate 'I am looking for the train station' into Italian and Mandarin.",
        "Explain the difference between 'their', 'there', and 'they're' with examples.",
        "What are false cognates? Give three examples between English and Spanish.",
        "Summarize the key themes of Gabriel Garcia Marquez's '100 Years of Solitude' in Spanish.",
    
        # --- Summarization & Writing (12 prompts) ---
        "Draft a short customer-support reply to a user who was billed twice for the same subscription.",
        "Outline a three-day study plan for preparing for a machine learning systems interview at a big tech company.",
        "Write a short product description for a reusable steel water bottle aimed at college students.",
        "Summarize the plot of Shakespeare's Macbeth in three bullet points.",
        "You are debugging a serving regression. Symptoms: p95 latency doubled after a model rollout, GPU utilization dropped, and request volume stayed flat. Summarize likely causes.",
        "Write an introductory email to a new client welcoming them to our financial advisory firm.",
        "Give me a 5-step recipe for making a traditional Italian Carbonara.",
        "Write a compelling conclusion paragraph for an essay arguing that remote work increases employee productivity.",
        "Write a professional email declining a job offer but keeping the door open for future opportunities.",
        "Summarize the history of the early public internet (1990-2000) in 150 words.",
        "Write a short, persuasive pitch for a new smartphone app that helps people track their daily water intake.",
        "Give me a beginner-friendly 3-day workout routine focusing on bodyweight exercises."
    ]
}

with open("robust_prompts.json", "w", encoding="utf-8") as f:
    json.dump(prompts, f, indent=2)

print(f"Generated robust_prompts.json with {len(prompts['prompts'])} prompts.")
