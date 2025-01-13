from flask import Flask, render_template, request, jsonify, session, send_file
import google.generativeai as genai
from gtts import gTTS
import base64
import re
from io import BytesIO
import requests
from bs4 import BeautifulSoup
import validators
import json
from urllib.parse import urlparse
import logging
from PIL import Image
from io import BytesIO
import concurrent.futures
import uuid
import os
import speech_recognition as sr
import wave
import PyPDF2
from docx import Document
import matplotlib.pyplot as plt
import numpy as np
import re
import base64
from io import BytesIO
import edge_tts
import asyncio

app = Flask(__name__)
app.secret_key = os.urandom(24)  # For session management

# Configure Gemini API
GOOGLE_API_KEY = "AIzaSyDMDAmWj3uGKOARGo-9iZT2_0RtKfg-tpA"  # Replace with your actual API key
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')
vision_model = genai.GenerativeModel('gemini-1.5-flash')  # Model for image analysis

# In-memory session context to maintain conversation
session_conversations = {}

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_function(input_str):
    """
    Chuyển đổi hàm số từ đầu vào của người dùng thành cú pháp Python hợp lệ.
    - Thay thế dấu ^ thành ** (mũ trong Python).
    - Tự động thêm dấu * giữa các hệ số và biến (ví dụ: 2x -> 2*x).
    - Tự động thêm dấu ngoặc cho phân số (ví dụ: 2x+1/3x-1 -> (2*x+1)/(3*x-1)).
    """
    # Thay thế dấu ^ thành **
    input_str = input_str.replace('^', '**')
    # Tự động thêm dấu * giữa các hệ số và biến (ví dụ: 2x -> 2*x)
    input_str = re.sub(r'(\d+)([a-zA-Z])', r'\1*\2', input_str)  # 2x -> 2*x
    input_str = re.sub(r'([a-zA-Z])(\d+)', r'\1*\2', input_str)  # x2 -> x*2
    # Tự động thêm dấu ngoặc cho phân số (ví dụ: 2x+1/3x-1 -> (2*x+1)/(3*x-1))
    if '/' in input_str:
        numerator, denominator = input_str.split('/', 1)
        input_str = f"({numerator})/({denominator})"
    return input_str

def plot_function(function_input):
    """
    Vẽ đồ thị hàm số dựa trên đầu vào của người dùng.@
    Trả về base64 của hình ảnh đồ thị.
    """
    # Chuyển đổi hàm số thành cú pháp Python
    parsed_function = parse_function(function_input)
    
    # Định nghĩa hàm số từ chuỗi đầu vào
    def f(x):
        return eval(parsed_function)
    
    # Tạo dữ liệu cho trục x
    x_values = np.linspace(-10, 10, 400)  # Từ -10 đến 10 với 400 điểm
    
    # Tính giá trị y tương ứng
    y_values = f(x_values)
    
    # Vẽ đồ thị
    plt.figure(figsize=(8, 6))
    plt.plot(x_values, y_values, label=f"f(x) = {function_input}", color='red')
    plt.axhline(0, color='black', linewidth=0.5)  # Trục Ox
    plt.axvline(0, color='black', linewidth=0.5)  # Trục Oy
    plt.title("Đồ thị hàm số")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.grid(True)
    plt.legend()
    
    # Lưu đồ thị vào buffer và chuyển thành base64
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    return f"data:image/png;base64,{image_base64}"

def clean_and_format_text(raw_text):
    """Clean and format text response from API, remove *."""
    lines = raw_text.split('\n')
    clean_lines = []
    for line in lines:
        clean_line = re.sub(r'^[\d.\-•]+\s*', '', line.strip())
        clean_line = clean_line.replace("*", "").strip()
        if clean_line:
             clean_lines.append(clean_line)
    return ' '.join(clean_lines)

def format_long_text(text, max_length=500):
    """Format long text by adding newlines."""
    words = text.split()
    formatted_text = ""
    line = ""
    for word in words:
        if len(line + word) + 1 <= max_length:
            line += (word + " ")
        else:
            formatted_text += line.strip() + "\n"
            line = word + " "
    formatted_text += line.strip()
    return formatted_text

def generate_response(question, context, image_parts=None, document_content=None, language="vi"):
    """Generate AI response based on question, context, optional image, and optional document,
    with language support."""
    try:
        if image_parts:
            # Combine text prompt with image parts for multi-modal input
            prompt_parts = [f"{context}\nNgười dùng: {question}\nAI: (Là LCFS Assistant, hỗ trợ học tập, hãy trả lời đầy đủ và chi tiết nhất có thể)"]
            prompt_parts.extend(image_parts)
            response = vision_model.generate_content(prompt_parts, stream=True)
            response_text = ""
            for chunk in response:
              response_text += chunk.text
            formatted_response = format_long_text(clean_and_format_text(response_text))
            return formatted_response
        elif document_content:
            # Generate response based on document content
            prompt = f"{context}\nDocument Content:\n{document_content}\nNgười dùng: {question}\nAI: (Là LCFS Assistant, hỗ trợ học tập, hãy trả lời đầy đủ và chi tiết nhất có thể)"
            response = model.generate_content(prompt)
            if response.text.strip():
                formatted_response = format_long_text(clean_and_format_text(response.text))
                return formatted_response
            else:
                return f"Xin lỗi, tôi không thể trả lời câu hỏi này một cách chi tiết dựa trên tài liệu đã cung cấp. Tuy nhiên, tôi có thể cung cấp một số thông tin cơ bản về chủ đề '{question}'."
        else:
            if language == "en":
                prompt = f"{context}\nUser: {question}\nAI: (As LCFS Assistant, a learning aid, please answer as fully and detailed as possible)"
            else:  # Default to Vietnamese
                prompt = f"{context}\nNgười dùng: {question}\nAI: (Là LCFS Assistant, hỗ trợ học tập, hãy trả lời đầy đủ và chi tiết nhất có thể)"

            response = model.generate_content(prompt)
            if response.text.strip():
                formatted_response = format_long_text(clean_and_format_text(response.text))
                return formatted_response
            else:
                if language == "en":
                    return f"Sorry, I can't answer this question in detail. However, I can provide some basic information about '{question}'."
                else:
                    return f"Xin lỗi, tôi không thể trả lời câu hỏi này một cách chi tiết. Tuy nhiên, tôi có thể cung cấp một số thông tin cơ bản về chủ đề '{question}'."
    except Exception as e:
        logging.error(f"Error generating response: {e}")
        if language == "en":
            return f"Sorry, I can't answer this question in detail. However, I can provide some basic information about '{question}'."
        else:
            return f"Xin lỗi, tôi không thể trả lời câu hỏi này một cách chi tiết. Tuy nhiên, tôi có thể cung cấp một số thông tin cơ bản về chủ đề '{question}'."

def shorten_url(url):
    """Shorten the URL to a readable format."""
    try:
        parsed_url = urlparse(url)
        base_domain = parsed_url.netloc
        if base_domain.startswith("www."):
             base_domain = base_domain[4:]
        path = parsed_url.path.strip("/")
        shortened_url = f"{base_domain}/{path}"
        return shortened_url if len(shortened_url) <= 50 else f"{base_domain}/..."

    except Exception as e:
        logging.error(f"Error shortening URL: {e}")
        return url

def fetch_image(url):
    """Fetches the image data from a URL and returns its base64 encoded string."""
    try:
        response = requests.get(url, stream=True, timeout=5)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content))
        image = image.convert('RGB')  # Ensure RGB
        image.thumbnail((150, 150))  # Resize while preserving aspect ratio
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return f"data:image/jpeg;base64,{img_str}"
    except Exception as e:
        logging.error(f"Error fetching or processing image {url}: {e}")
        return None

def extract_images_from_html(soup):
    """Extracts image URLs from HTML content, prioritizing og:image tag"""
    og_image = soup.find("meta", property="og:image")
    if og_image and og_image.has_attr("content"):
        return [og_image['content']]
    
    images = soup.find_all('img', limit=3)
    return [img['src'] for img in images if img.has_attr('src')]

def search_and_summarize(query):
    """Search the web and provide a focused summary using BeautifulSoup and AI with image thumbnails."""
    search_query = query.replace(" ", "+")
    url = f"https://www.google.com/search?q={search_query}"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        search_results = soup.find_all('div', class_='yuRUbf')[:3]
        results = []
        combined_content = ""
        search_result_images = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = []

            for result in search_results:
                link_tag = result.find('a')
                if link_tag and link_tag.has_attr('href'):
                    link = link_tag['href']
                    future = executor.submit(process_link, link, query, headers)
                    futures.append(future)
        
            for future in concurrent.futures.as_completed(futures):
                try:
                    res = future.result()
                    if res:
                       results.append(res)
                       combined_content += f"{res['title']}. {res['summary']}\n"
                       if res['image']:
                          search_result_images.append(res['image'])
                except Exception as e:
                    logging.error(f"Error processing link during search: {e}")

        # Use Gemini to summarize combined content
        if combined_content:
            prompt = f"Tóm tắt thông tin sau đây một cách chuyên nghiệp và gọn gàng, đúng trọng tâm, không lan man và cung cấp nguồn tham khảo: {combined_content}\nAI:"
            ai_summary = model.generate_content(prompt).text
            final_response = f"{format_long_text(clean_and_format_text(ai_summary))}\n\nNguồn: "
            
            # Modify source display to show full links with sources
            sources_part = ", ".join([f"{r['source']}: {r['link']}" for r in results])
            final_response += sources_part
            
            # Add the thumbnail images
            formatted_results = []
            for r in results:
               formatted_results.append({
                    "title": r["title"],
                    "link": r["link"],
                    "summary": r["summary"],
                    "source": r["source"],
                    "image": r.get("image", None)
                })

            return {
                    "summary": final_response,
                    "results": formatted_results,
                    "images": search_result_images
                }
        else:
            return "Không tìm thấy thông tin phù hợp."
 
    except requests.exceptions.RequestException as e:
        logging.error(f"Error during search request: {e}")
        return f"Không thể tìm kiếm thông tin do lỗi: {str(e)}"
    
def process_link(link, query, headers):
    try:
       link_response = requests.get(link, headers=headers, timeout=10)
       link_response.raise_for_status()
       link_soup = BeautifulSoup(link_response.text, 'html.parser')

       # Extract title
       title_tag = link_soup.find('title')
       title = title_tag.text.strip() if title_tag else "Không có tiêu đề"

       # Find relevant content based on the query
       paragraphs = link_soup.find_all('p')
       relevant_sentences = []
       for p in paragraphs:
           sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', p.text)
           for sentence in sentences:
                if query.lower() in sentence.lower():
                    relevant_sentences.append(sentence.strip())

       summary = " ".join(relevant_sentences)
       if not summary:
             paragraphs = link_soup.find_all('p')[:2]
             summary = " ".join([p.text for p in paragraphs])

       # Extract domain for source
       domain = link.split('//')[-1].split('/')[0]
       if domain.startswith("www."):
         domain = domain[4:]
    
       # Extract and process image thumbnail
       image_urls = extract_images_from_html(link_soup)
       image_url = image_urls[0] if image_urls else None
       image_base64 = fetch_image(image_url) if image_url else None

       return {
           "title": title,
           "link": link,
           "summary": clean_and_format_text(summary),
           "source": domain,
           "image": image_base64
        }

    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching or processing link {link}: {e}")
        return {
            "title": "Không thể truy cập trang web.",
            "link": link,
            "summary": "Không thể truy cập trang web.",
            "source": "Không xác định"
        }
    except Exception as e:
       logging.error(f"An unexpected error occurred while processing link {link}: {e}")
       return {
           "title": "Đã xảy ra lỗi.",
            "link": link,
            "summary": "Đã xảy ra lỗi.",
            "source": "Không xác định"
        }

def process_images_for_request(image_data_list):
    """Processes a list of base64 image data and returns a list of image parts for the Gemini API."""
    image_parts = []
    for image_data in image_data_list:
        try:
            # Decode base64 image data
            image_bytes = base64.b64decode(image_data.split(',')[1])
            image = Image.open(BytesIO(image_bytes))
            
            # Ensure the image is in RGB format
            if image.mode != "RGB":
                image = image.convert("RGB")
            
            # Resize the image (optional, but recommended for faster processing)
            max_size = (1024, 1024)
            if image.width > max_size[0] or image.height > max_size[1]:
                image.thumbnail(max_size)
            
            # Convert the image to a format suitable for the Gemini API
            buffered = BytesIO()
            image.save(buffered, format="JPEG")  # or PNG, depending on the model's preference
            mime_type = "image/jpeg"  # or image/png
            
            image_parts.append({
                "mime_type": mime_type,
                "data": base64.b64encode(buffered.getvalue()).decode("utf-8")
            })
        except Exception as e:
            logging.error(f"Error processing image: {e}")
            # Handle the error as appropriate for your application
            # You might want to return a specific error message or skip the image
    return image_parts

def read_pdf(file):
    """Reads and returns the text content of a PDF file."""
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        logging.error(f"Error reading PDF: {e}")
        return None

def read_docx(file):
    """Reads and returns the text content of a DOCX file."""
    try:
        doc = Document(file)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    except Exception as e:
        logging.error(f"Error reading DOCX: {e}")
        return None

def generate_conversation_title(messages):
    """Generate a title for the conversation using the first few messages."""
    combined_text = " ".join([msg["content"] for msg in messages if msg["role"] == "user"])
    if combined_text:
        prompt = f"Tạo một tiêu đề ngắn gọn (dưới 5 từ) cho cuộc trò chuyện này: {combined_text[:200]}"
        try:
            response = model.generate_content(prompt)
            title = response.text.strip()
            title = re.sub(r'[\\/*?:"<>|]', "", title)
            return title
        except Exception as e:
            logging.error(f"Error generating conversation title: {e}")
            return "Cuộc trò chuyện mới"
    else:
        return "Cuộc trò chuyện mới"
    
def get_conversation_history(session_id):
    """Get the conversation history for the given session ID."""
    return session_conversations.get(session_id, [])

def add_message_to_conversation(session_id, role, content, image_data_list=None, response_type=None, response_content=None, document_data=None):
    """Add a message to the conversation history."""
    if session_id not in session_conversations:
        session_conversations[session_id] = []

    message = {
        "role": role,
        "content": content,
        "images": image_data_list or [],
        "response_type": response_type
    }
    
    # Store additional details for 'realtime' type
    if response_type == 'realtime' and response_content:
        message["realtime_details"] = {
            "summary": response_content.get("summary", ""),
            "results": response_content.get("results", []),
            "images": response_content.get("images", [])
        }
    
    # Store document data if available
    if document_data:
        message["document"] = document_data

    session_conversations[session_id].append(message)

# Function to convert speech to text
def convert_speech_to_text():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Xin mời nói...")
        audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
        print("Đang xử lý giọng nói...")

        # Save the audio to a temporary file
        temp_filename = f"temp_audio_{uuid.uuid4()}.wav"
        with wave.open(temp_filename, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # Assuming 16-bit PCM
            wf.setframerate(44100)  # Assuming 44.1kHz sample rate
            wf.writeframes(audio.frame_data)

    try:
        text = recognizer.recognize_google(audio, language="vi-VN")
        print(f"Bạn đã nói: {text}")
        return text, temp_filename
    except sr.UnknownValueError:
        print("Không nhận dạng được giọng nói")
        os.remove(temp_filename)  # Remove the temporary audio file
        return "Không nhận dạng được giọng nói", None
    except sr.RequestError as e:
        print(f"Lỗi kết nối: {e}")
        os.remove(temp_filename)  # Remove the temporary audio file
        return f"Lỗi kết nối: {e}", None
    except Exception as e:
        print(f"Lỗi: {e}")
        os.remove(temp_filename)  # Remove the temporary audio file
        return f"Lỗi: {e}", None

@app.route('/')
def home():
    """Render main interface."""
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    return render_template('index.html')

@app.route('/api/get_conversations', methods=['GET'])
def get_conversations():
    """Retrieve all conversation histories."""
    conversations_with_titles = []
    for session_id, messages in session_conversations.items():
        title = generate_conversation_title(messages)
        conversations_with_titles.append({
            "session_id": session_id,
            "title": title
        })
    return jsonify(conversations_with_titles)

@app.route('/api/load_conversation', methods=['GET'])
def load_conversation():
    """Load a specific conversation."""
    session_id = request.args.get('session_id')
    conversation = get_conversation_history(session_id)
    return jsonify(conversation)

@app.route('/api/new_conversation', methods=['POST'])
def new_conversation():
    """Start a new conversation."""
    session['session_id'] = str(uuid.uuid4())
    return jsonify({"message": "New conversation started", "session_id": session['session_id']})

async def generate_audio_edge_tts(text, voice):
    """Generate audio using edge-tts."""
    try:
        communicate = edge_tts.Communicate(text, voice)
        audio_stream = BytesIO()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_stream.write(chunk["data"])
        audio_stream.seek(0)
        audio_base64 = base64.b64encode(audio_stream.read()).decode('utf-8')
        return audio_base64
    except Exception as e:
        logging.error(f"Error generating audio with edge-tts: {e}")
        return None

@app.route('/api/respond', methods=['POST'])
def respond():
    """API to handle user queries, image analysis, and document analysis."""
    session_id = session.get('session_id', str(uuid.uuid4()))
    data = request.json
    question = data.get("question", "").strip()
    response_type = data.get("response_type", "text")
    selected_voice = data.get("selected_voice", "gtts")  # Get the selected voice
    image_data_list = data.get("image_data", [])
    document_data = data.get("document_data")  # New: Get document data
    audio_response_requested = data.get("audio_response", False)

    # Determine the language based on the selected voice
    if selected_voice.startswith("en-"):
        language = "en"
    else:
        language = "vi"

    if not question and not image_data_list and not document_data:
        return jsonify({"error": "Chưa nhập câu hỏi, tải lên hình ảnh hoặc tài liệu"}), 400

    try:
        # Process images if any are provided
        image_parts = process_images_for_request(image_data_list)

        # Process document if provided
        document_content = None
        if document_data:
            try:
                file_content = base64.b64decode(document_data["content"])
                if document_data["type"] == "application/pdf":
                    document_content = read_pdf(BytesIO(file_content))
                elif document_data["type"] == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                    document_content = read_docx(BytesIO(file_content))
                else:
                    return jsonify({"error": "Unsupported file type"}), 400
            except Exception as e:
                logging.error(f"Error processing document: {e}")
                return jsonify({"error": "Error processing document"}), 500

        # Generate response based on whether images or document were provided, and the selected language
        if image_parts:
            response = generate_response(question, "", image_parts, language=language)
            if not question:
              question = "Phân tích hình ảnh" if language == "vi" else "Analyze the image"
            # Store the question, response, and images in the conversation history
            add_message_to_conversation(session_id, "user", question, image_data_list, response_type)
            add_message_to_conversation(session_id, "ai", response, response_type=response_type)
        elif document_content:
            response = generate_response(question, "", document_content=document_content, language=language)
            add_message_to_conversation(session_id, "user", question, response_type=response_type, document_data=document_data)
            add_message_to_conversation(session_id, "ai", response, response_type=response_type)
        else:
            context = " ".join([f"{entry['role']}: {entry['content']}"
                                for entry in get_conversation_history(session_id)])
            if response_type == "realtime":
                response = search_and_summarize(question)
                # Extract only the summary part for conversation history
                response_text = response.get("summary", response) if isinstance(response, dict) else response
                add_message_to_conversation(session_id, "user", question, response_type=response_type)
                add_message_to_conversation(session_id, "ai", response_text, response_type=response_type, response_content=response)
                return jsonify({"type": "realtime", "content": response, "session_id": session_id})
            else:
                response = generate_response(question, context, language=language)
                add_message_to_conversation(session_id, "user", question, response_type=response_type)
                add_message_to_conversation(session_id, "ai", response, response_type=response_type)

        # Handle different response types
        if response_type == "text":
            if audio_response_requested:
                if selected_voice == "gtts":
                    tts = gTTS(text=response, lang=language, slow=False)
                    audio_stream = BytesIO()
                    tts.write_to_fp(audio_stream)
                    audio_stream.seek(0)
                    audio_base64 = base64.b64encode(audio_stream.read()).decode('utf-8')
                else:
                    # Use edge-tts for other voices
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    audio_base64 = loop.run_until_complete(generate_audio_edge_tts(response, selected_voice))
                    loop.close()
                
                if audio_base64:
                    return jsonify({"type": "audio_text", "content": {"text": response, "audio": audio_base64}, "session_id": session_id})
                else:
                    return jsonify({"error": "Failed to generate audio"}), 500
            else:
                return jsonify({"type": "text", "content": response, "session_id": session_id})

        elif response_type == "audio":
            if selected_voice == "gtts":
                tts = gTTS(text=response, lang=language, slow=False)
                audio_stream = BytesIO()
                tts.write_to_fp(audio_stream)
                audio_stream.seek(0)
                audio_base64 = base64.b64encode(audio_stream.read()).decode('utf-8')
            else:
                # Use edge-tts for other voices
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                audio_base64 = loop.run_until_complete(generate_audio_edge_tts(response, selected_voice))
                loop.close()

            if audio_base64:
                return jsonify({"type": "audio", "content": {"text": response, "audio": audio_base64}, "session_id": session_id})
            else:
                return jsonify({"error": "Failed to generate audio"}), 500

        elif response_type == "mindmap":
            sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', response)
            main_idea = question
            sub_ideas = []
            level1_ideas = {}

            for i, sentence in enumerate(sentences):
                sentence = sentence.strip()
                if not sentence:
                    continue

                if i % 4 == 0:
                    level1_ideas[sentence] = []
                    sub_ideas.append({"text": sentence})
                else:
                    current_level1 = list(level1_ideas.keys())[-1]
                    level1_ideas[current_level1].append(sentence)
                    sub_ideas.append({"text": sentence, "parent": current_level1})

            mindmap_data = {
                "mainIdea": main_idea,
                "subIdeas": sub_ideas
            }
            return jsonify({"type": "mindmap", "content": mindmap_data, "session_id": session_id})

        else:
            return jsonify({"type": "text", "content": response, "session_id": session_id})

    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        return jsonify({"error": str(e)}), 500

        
@app.route('/api/plot_graph', methods=['POST'])
def plot_graph():
    """
    API endpoint để vẽ đồ thị hàm số.
    """
    data = request.json
    function_input = data.get("function_input", "").strip()
    
    if not function_input:
        return jsonify({"error": "Vui lòng nhập hàm số"}), 400
    
    try:
        image_base64 = plot_function(function_input)
        return jsonify({"type": "image", "content": image_base64})
    except Exception as e:
        logging.error(f"Error plotting graph: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/speech_to_text', methods=['POST'])
def speech_to_text():
    """API endpoint to convert speech to text."""
    text, temp_filename = convert_speech_to_text()
    if temp_filename:
        return jsonify({"text": text, "temp_filename": temp_filename})
    else:
        return jsonify({"error": text}), 400
    
@app.route('/download_audio/<filename>')
def download_audio(filename):
    """Serve the temporary audio file for download."""
    try:
        return send_file(filename, as_attachment=True)
    except Exception as e:
        logging.error(f"Error serving audio file: {e}")
        return "Error serving audio file", 500
    finally:
        # Clean up the temporary file after serving
        if os.path.exists(filename):
            os.remove(filename)

if __name__ == '__main__':
    app.run(debug=True)

