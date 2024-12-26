
A FastAPI-based API using scikit-learn to classify text inputs. It preprocesses text (e.g., TF-IDF), trains a model (e.g., SVM), and provides a /predict/ endpoint to return categories and confidence scores. Ideal for tasks such as sentiment analysis and spam detection.

Project Structure app/: Contains the FastAPI app and API routes. model/: Includes model training and prediction logic. requirements.txt: Lists dependencies. README.md: Documentation. How to Run Install Dependencies: Run pip install -r requirements.txt to install the required packages.

Train the Model: Execute python model/train.py to preprocess text and train the SVM model.

Start the API: Use uvicorn app.main:app --reload to launch the FastAPI server.
