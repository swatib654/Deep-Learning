import speech_recognition as sr
import pyttsx3

# Initialize recognizer and TTS engine
recognizer = sr.Recognizer()
engine = pyttsx3.init()

def speak(text):
    """Convert text to speech"""
    engine.say(text)
    engine.runAndWait()

def listen_command():
    """Listen for voice input and return recognized text"""
    with sr.Microphone() as source:
        print("listening....")
        audio = recognizer.listen(source)
        try:
            command = recognizer.recognize_google(audio)
            print("Received command:", command)
            return command.lower()
        except sr.UnknownValueError:
            print("Sorry, could not understand the audio.")
            speak("Sorry, I could not understand.")
            return ""
        except sr.RequestError:
            print("Speech service unavailable.")
            speak("Speech service is not available.")
            return ""

if __name__ == "__main__":
    command = listen_command()
    if command:
        speak(f"You said {command}")
