Name:-Piyush Agarwal
Contact:- email:-piyushagarwal14.pa@gmail.com

Loom video:-https://www.loom.com/share/ac568af19d2146f5a4e1844ea58decb1?sid=8e6854c0-a944-44e5-a211-f005ca0794b3
Streamlit app link:-https://llm-assignment-piyushagarwal.streamlit.app
 

PDF Chatbot

PDF Chatbot is a Streamlit-based web application that allows users to upload PDF and text files, interact with a chatbot, and ask questions based on the content of the uploaded documents. The chatbot leverages Google's Generative AI model to provide responses to user queries.

Features:-
1)Upload PDF and text files: Users can upload PDF and text files containing textual content for the chatbot to analyze.

2)Interactive chat interface: Users can ask questions to the chatbot based on the content of the uploaded documents.

3)Real-time responses: The chatbot provides real-time responses to user queries, facilitating interactive conversations.

4)Conversational memory: The chatbot maintains a conversational memory to provide contextually relevant responses.

Technical Details:-
Technologies Used:-
1)Python: The backend logic of the application is written in Python.

2)Streamlit: Streamlit is used to create the web interface for the application.

3)PyPDF2: PyPDF2 is used to extract text from PDF files.

4)Google's Generative AI: Google's Generative AI model is used to power the chatbot's conversational capabilities.

5)MongoDB: MongoDB is used as the database to store document text for retrieval during conversations.

Architecture:-
The application consists of two main components:

1)Backend: The backend logic is responsible for processing user uploads, extracting text from documents, interacting with the chatbot, and storing document text in MongoDB.

2)Frontend: The frontend, built using Streamlit, provides the user interface for uploading documents, interacting with the chatbot, and displaying chat messages.

Workflow:-
1)Document Upload: Users upload PDF or text files containing textual content.
2)Text Extraction: The backend extracts text from the uploaded documents using PyPDF2.
3)Chatbot Interaction: Users can ask questions to the chatbot based on the content of the uploaded documents.
4)Response Generation: The chatbot generates responses using Google's Generative AI model (Gemini Pro 001).
5)Display Response: The response generated by the chatbot is displayed to the user in real-time.
6)Conversational Memory: The chatbot maintains a conversational memory to provide contextually relevant responses.
