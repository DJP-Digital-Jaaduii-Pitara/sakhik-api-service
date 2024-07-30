import os
import time
from logger import logger
import re
from env_manager import translate_class as translator
from utils import get_from_env_or_config

DEFAULT_LANGAUGE = get_from_env_or_config('default', 'language', None)
TRANSLATION_TYPE = os.getenv('TRANSLATION_TYPE')
BUCKET_TYPE = os.getenv('BUCKET_TYPE')

def process_incoming_voice(file_url, input_language):
    """
    Main Function for processing audio based queries
    """
    error_message = None
    if not TRANSLATION_TYPE:
        return None, None, "Translation service is not configured!"

    try:
        regional_text = translator.speech_to_text(file_url, input_language)
        try:
            english_text = translator.translate_text(text=regional_text, source=input_language, destination=DEFAULT_LANGAUGE)
        except Exception as e:
            error_message = "Indic translation to English failed"
            logger.error(f"Exception occurred: {e}", exc_info=True)
            english_text = None
    except Exception as e:
        error_message = "Speech to text conversion API failed"
        logger.error(f"Exception occurred: {e}", exc_info=True)
        regional_text = None
        english_text = None
    return regional_text, english_text, error_message


def process_incoming_text(regional_text, input_language):
    """
    Main function for processing text queries
    """
    error_message = None
    if not TRANSLATION_TYPE:
        return regional_text, error_message

    try:
        english_text = translator.translate_text(text=regional_text, source=input_language, destination=DEFAULT_LANGAUGE)
    except Exception as e:
        error_message = "Indic translation to English failed"
        english_text = None
        logger.error(f"Exception occurred: {e}", exc_info=True)
    return english_text, error_message


def process_outgoing_text(english_text, input_language):
    """
    Main func for generating text response
    """
    error_message = None
    if not TRANSLATION_TYPE:
        return english_text, error_message

    sections = split_text(english_text)
    translated_sections, error_message = translate_sections(sections, input_language)
    if error_message is not None:
        return None, error_message
    formatted_translated_text = reassemble_text(translated_sections)
    return formatted_translated_text, None

def split_text(text):
    return re.split(r'(\t|\n)', text)


def reassemble_text(sections):
    return ''.join(sections)


def translate_sections(sections, input_language):
    translated_sections = []
    error_message = None
    for section in sections:
        if section.strip():
            try:
                translated_text = translator.translate_text(text=section, source=DEFAULT_LANGAUGE, destination=input_language)
            except Exception as e:
                error_message = "English translation to indic language failed"
                logger.error(f"Exception occurred: {e}", exc_info=True)
                translated_text = None
                break
            translated_sections.append(translated_text)
        else:
            translated_sections.append(section)
    return translated_sections, error_message

def process_outgoing_voice(message, input_language):
    """
    Main function for generating audio response
    """
    if not BUCKET_TYPE:
        return None, "Storage service is not configured!"

    error_message = None
    decoded_audio_content = translator.text_to_speech(language=input_language, text=message)
    if decoded_audio_content is not None:
        logger.info("Creating output MP3 file")
        time_stamp = time.strftime("%Y%m%d-%H%M%S")
        filename = "audio-output-" + time_stamp + ".mp3"
        output_mp3_file = open(filename, "wb")
        output_mp3_file.write(decoded_audio_content)
        logger.info("Audio Response is saved as a MP3 file.")
        return output_mp3_file, error_message
    error_message = "Text to Audio conversion failed"
    logger.error(error_message)
    return None, error_message
