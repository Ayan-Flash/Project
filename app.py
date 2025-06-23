import os
import logging
import tempfile
import shutil
import zipfile
import json
import time
import random
from flask import Flask, render_template, request, jsonify, send_file, flash, redirect, url_for
from pytubefix import YouTube, Playlist
from moviepy import VideoFileClip, AudioFileClip
import re
from urllib.parse import urlparse, parse_qs
import threading
from collections import defaultdict

logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev-secret-key")

# Create downloads directory if it doesn't exist
DOWNLOADS_DIR = os.path.join(os.getcwd(), 'downloads')
os.makedirs(DOWNLOADS_DIR, exist_ok=True)

# Store batch download progress
batch_progress = defaultdict(dict)
batch_lock = threading.Lock()

def is_valid_youtube_url(url):
    """Validate YouTube URL"""
    youtube_regex = re.compile(
        r'(https?://)?(www\.)?(youtube|youtu|youtube-nocookie)\.(com|be)/'
        r'(watch\?v=|embed/|v/|.+\?v=)?([^&=%\?]{11})'
    )
    playlist_regex = re.compile(
        r'(https?://)?(www\.)?youtube\.com/playlist\?list=([a-zA-Z0-9_-]+)'
    )
    return youtube_regex.match(url) is not None or playlist_regex.match(url) is not None

def create_youtube_object(url, max_retries=3):
    """Create YouTube object with retry logic to handle 400 errors"""
    for attempt in range(max_retries):
        try:
            # Add small delay between retries
            if attempt > 0:
                time.sleep(random.uniform(1, 3))
            
            # pytubefix has better handling of YouTube's anti-bot measures
            yt = YouTube(url)
            
            # Test access to trigger any potential errors
            _ = yt.title
            return yt
            
        except Exception as e:
            logging.warning(f"Attempt {attempt + 1} failed for {url}: {str(e)}")
            if attempt == max_retries - 1:
                raise e
    
    return None

def is_playlist_url(url):
    """Check if URL is a playlist"""
    return 'playlist' in url and 'list=' in url

def get_video_id(url):
    """Extract video ID from YouTube URL"""
    parsed_url = urlparse(url)
    if parsed_url.hostname == 'youtu.be':
        return parsed_url.path[1:]
    if parsed_url.hostname in ('www.youtube.com', 'youtube.com'):
        if parsed_url.path == '/watch':
            return parse_qs(parsed_url.query)['v'][0]
        if parsed_url.path[:7] == '/embed/':
            return parsed_url.path.split('/')[2]
        if parsed_url.path[:3] == '/v/':
            return parsed_url.path.split('/')[2]
    return None

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/get_video_info', methods=['POST'])
def get_video_info():
    """Get video information and available qualities"""
    try:
        url = request.form.get('url', '').strip()
        
        if not url:
            return jsonify({'error': 'Please provide a YouTube URL'}), 400
            
        if not is_valid_youtube_url(url):
            return jsonify({'error': 'Please provide a valid YouTube URL'}), 400
        
        # Check if it's a playlist
        if is_playlist_url(url):
            return get_playlist_info(url)
        
        # Create YouTube object with retry logic
        yt = create_youtube_object(url)
        if not yt:
            return jsonify({'error': 'Failed to access video. Video may be private, restricted, or unavailable.'}), 400
        
        # Get available video streams
        video_streams = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc()
        
        # Extract quality options
        qualities = []
        for stream in video_streams:
            if stream.resolution:
                qualities.append({
                    'itag': stream.itag,
                    'quality': stream.resolution,
                    'filesize': stream.filesize
                })
        
        # Remove duplicates and keep only common qualities
        unique_qualities = []
        seen_resolutions = set()
        for quality in qualities:
            if quality['quality'] not in seen_resolutions:
                unique_qualities.append(quality)
                seen_resolutions.add(quality['quality'])
        
        return jsonify({
            'type': 'video',
            'title': yt.title,
            'length': yt.length,
            'thumbnail': yt.thumbnail_url,
            'qualities': unique_qualities,
            'has_audio': len(yt.streams.filter(only_audio=True)) > 0
        })
        
    except Exception as e:
        logging.error(f"Error getting video info: {str(e)}")
        return jsonify({'error': f'Error fetching video information: {str(e)}'}), 500

def get_playlist_info(url):
    """Get playlist information"""
    try:
        playlist = Playlist(url)
        videos = []
        
        for i, video_url in enumerate(playlist.video_urls[:10]):  # Limit to first 10 for preview
            try:
                yt = create_youtube_object(video_url)
                if not yt:
                    continue
                videos.append({
                    'url': video_url,
                    'title': yt.title,
                    'length': yt.length,
                    'thumbnail': yt.thumbnail_url
                })
            except Exception as e:
                logging.warning(f"Error getting info for video {i}: {str(e)}")
                continue
        
        return jsonify({
            'type': 'playlist',
            'title': playlist.title,
            'total_videos': len(playlist.video_urls),
            'videos': videos
        })
        
    except Exception as e:
        logging.error(f"Error getting playlist info: {str(e)}")
        return jsonify({'error': f'Error fetching playlist information: {str(e)}'}), 500

@app.route('/download_video', methods=['POST'])
def download_video():
    """Download video in selected quality"""
    try:
        url = request.form.get('url', '').strip()
        itag = request.form.get('itag')
        
        if not url or not itag:
            return jsonify({'error': 'Missing URL or quality selection'}), 400
            
        if not is_valid_youtube_url(url):
            return jsonify({'error': 'Invalid YouTube URL'}), 400
        
        # Create YouTube object with retry logic
        yt = create_youtube_object(url)
        if not yt:
            return jsonify({'error': 'Failed to access video. Video may be private, restricted, or unavailable.'}), 400
        
        # Get the selected stream
        stream = yt.streams.get_by_itag(int(itag))
        if not stream:
            return jsonify({'error': 'Selected quality not available'}), 400
        
        # Generate safe filename
        safe_title = re.sub(r'[^\w\s-]', '', yt.title).strip()
        safe_title = re.sub(r'[-\s]+', '-', safe_title)
        filename = f"{safe_title}_{stream.resolution}.mp4"
        
        # Download to temporary location first
        temp_dir = tempfile.mkdtemp()
        temp_file = stream.download(output_path=temp_dir, filename=filename)
        
        return send_file(
            temp_file,
            as_attachment=True,
            download_name=filename,
            mimetype='video/mp4'
        )
        
    except Exception as e:
        logging.error(f"Error downloading video: {str(e)}")
        return jsonify({'error': f'Error downloading video: {str(e)}'}), 500

@app.route('/download_audio', methods=['POST'])
def download_audio():
    """Download video as audio in specified format"""
    try:
        url = request.form.get('url', '').strip()
        audio_format = request.form.get('format', 'mp3').lower()
        
        if not url:
            return jsonify({'error': 'Missing URL'}), 400
            
        if not is_valid_youtube_url(url):
            return jsonify({'error': 'Invalid YouTube URL'}), 400
        
        if audio_format not in ['mp3', 'wav', 'flac']:
            return jsonify({'error': 'Unsupported audio format'}), 400
        
        # Create YouTube object with retry logic
        yt = create_youtube_object(url)
        if not yt:
            return jsonify({'error': 'Failed to access video. Video may be private, restricted, or unavailable.'}), 400
        
        # Get audio stream
        audio_stream = yt.streams.filter(only_audio=True).first()
        if not audio_stream:
            return jsonify({'error': 'No audio stream available'}), 400
        
        # Generate safe filename
        safe_title = re.sub(r'[^\w\s-]', '', yt.title).strip()
        safe_title = re.sub(r'[-\s]+', '-', safe_title)
        
        # Create temporary directory
        temp_dir = tempfile.mkdtemp()
        
        # Download audio file
        temp_audio_file = audio_stream.download(output_path=temp_dir, filename='temp_audio')
        
        # Convert to requested format
        audio_filename = f"{safe_title}.{audio_format}"
        audio_path = os.path.join(temp_dir, audio_filename)
        
        # Use moviepy to convert audio
        video_clip = VideoFileClip(temp_audio_file)
        if video_clip.audio is not None:
            if audio_format == 'mp3':
                video_clip.audio.write_audiofile(audio_path, verbose=False, logger=None)
            elif audio_format == 'wav':
                video_clip.audio.write_audiofile(audio_path, verbose=False, logger=None, codec='pcm_s16le')
            elif audio_format == 'flac':
                video_clip.audio.write_audiofile(audio_path, verbose=False, logger=None, codec='flac')
            video_clip.close()
        else:
            video_clip.close()
            return jsonify({'error': 'No audio track found in video'}), 400
        
        # Clean up temp audio file
        os.remove(temp_audio_file)
        
        # Set appropriate MIME type
        mime_types = {
            'mp3': 'audio/mpeg',
            'wav': 'audio/wav',
            'flac': 'audio/flac'
        }
        
        return send_file(
            audio_path,
            as_attachment=True,
            download_name=audio_filename,
            mimetype=mime_types[audio_format]
        )
        
    except Exception as e:
        logging.error(f"Error downloading audio: {str(e)}")
        return jsonify({'error': f'Error downloading audio: {str(e)}'}), 500

@app.route('/download_playlist', methods=['POST'])
def download_playlist():
    """Download entire playlist"""
    try:
        url = request.form.get('url', '').strip()
        download_type = request.form.get('type', 'video')  # video or audio
        audio_format = request.form.get('format', 'mp3').lower()
        
        if not url:
            return jsonify({'error': 'Missing URL'}), 400
            
        if not is_valid_youtube_url(url) or not is_playlist_url(url):
            return jsonify({'error': 'Invalid playlist URL'}), 400
        
        # Generate batch ID
        import uuid
        batch_id = str(uuid.uuid4())
        
        # Start download in background thread
        thread = threading.Thread(
            target=process_playlist_download,
            args=(batch_id, url, download_type, audio_format)
        )
        thread.daemon = True
        thread.start()
        
        return jsonify({'batch_id': batch_id})
        
    except Exception as e:
        logging.error(f"Error starting playlist download: {str(e)}")
        return jsonify({'error': f'Error starting playlist download: {str(e)}'}), 500

@app.route('/batch_progress/<batch_id>')
def get_batch_progress(batch_id):
    """Get progress of batch download"""
    with batch_lock:
        progress = batch_progress.get(batch_id, {})
    return jsonify(progress)

@app.route('/download_batch/<batch_id>')
def download_batch_zip(batch_id):
    """Download completed batch as ZIP file"""
    try:
        with batch_lock:
            progress = batch_progress.get(batch_id, {})
        
        if not progress or progress.get('status') != 'completed':
            return jsonify({'error': 'Batch not ready for download'}), 400
        
        zip_path = progress.get('zip_path')
        if not zip_path or not os.path.exists(zip_path):
            return jsonify({'error': 'Download file not found'}), 404
        
        return send_file(
            zip_path,
            as_attachment=True,
            download_name=f"youtube_batch_{batch_id}.zip",
            mimetype='application/zip'
        )
        
    except Exception as e:
        logging.error(f"Error downloading batch: {str(e)}")
        return jsonify({'error': f'Error downloading batch: {str(e)}'}), 500

def process_playlist_download(batch_id, url, download_type, audio_format='mp3'):
    """Process playlist download in background"""
    try:
        with batch_lock:
            batch_progress[batch_id] = {
                'status': 'starting',
                'current': 0,
                'total': 0,
                'current_video': '',
                'errors': []
            }
        
        playlist = Playlist(url)
        total_videos = len(playlist.video_urls)
        
        with batch_lock:
            batch_progress[batch_id]['total'] = total_videos
            batch_progress[batch_id]['status'] = 'downloading'
        
        # Create temporary directory for batch
        temp_dir = tempfile.mkdtemp()
        downloaded_files = []
        
        for i, video_url in enumerate(playlist.video_urls):
            try:
                with batch_lock:
                    batch_progress[batch_id]['current'] = i + 1
                
                yt = create_youtube_object(video_url)
                if not yt:
                    continue
                safe_title = re.sub(r'[^\w\s-]', '', yt.title).strip()
                safe_title = re.sub(r'[-\s]+', '-', safe_title)
                
                with batch_lock:
                    batch_progress[batch_id]['current_video'] = yt.title
                
                if download_type == 'video':
                    # Download video
                    stream = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
                    if stream:
                        filename = f"{safe_title}.mp4"
                        file_path = stream.download(output_path=temp_dir, filename=filename)
                        downloaded_files.append(file_path)
                else:
                    # Download audio
                    audio_stream = yt.streams.filter(only_audio=True).first()
                    if audio_stream:
                        temp_audio = audio_stream.download(output_path=temp_dir, filename=f'temp_audio_{i}')
                        
                        # Convert to requested format
                        audio_filename = f"{safe_title}.{audio_format}"
                        audio_path = os.path.join(temp_dir, audio_filename)
                        
                        video_clip = VideoFileClip(temp_audio)
                        if video_clip.audio is not None:
                            if audio_format == 'mp3':
                                video_clip.audio.write_audiofile(audio_path, verbose=False, logger=None)
                            elif audio_format == 'wav':
                                video_clip.audio.write_audiofile(audio_path, verbose=False, logger=None, codec='pcm_s16le')
                            elif audio_format == 'flac':
                                video_clip.audio.write_audiofile(audio_path, verbose=False, logger=None, codec='flac')
                            video_clip.close()
                            os.remove(temp_audio)
                            downloaded_files.append(audio_path)
                        else:
                            video_clip.close()
                            os.remove(temp_audio)
                            
            except Exception as e:
                with batch_lock:
                    batch_progress[batch_id]['errors'].append(f"Error downloading {video_url}: {str(e)}")
                logging.error(f"Error downloading video {i}: {str(e)}")
                continue
        
        # Create ZIP file
        with batch_lock:
            batch_progress[batch_id]['status'] = 'creating_zip'
        
        zip_path = os.path.join(temp_dir, f'batch_{batch_id}.zip')
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for file_path in downloaded_files:
                if os.path.exists(file_path):
                    zip_file.write(file_path, os.path.basename(file_path))
        
        # Clean up individual files
        for file_path in downloaded_files:
            if os.path.exists(file_path):
                os.remove(file_path)
        
        with batch_lock:
            batch_progress[batch_id]['status'] = 'completed'
            batch_progress[batch_id]['zip_path'] = zip_path
            
    except Exception as e:
        with batch_lock:
            batch_progress[batch_id]['status'] = 'error'
            batch_progress[batch_id]['error'] = str(e)
        logging.error(f"Error in playlist download: {str(e)}")

@app.errorhandler(404)
def not_found(error):
    return render_template('index.html'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('index.html'), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)