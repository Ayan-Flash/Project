// YouTube Downloader JavaScript

document.addEventListener('DOMContentLoaded', function() {
    const urlForm = document.getElementById('urlForm');
    const videoUrlInput = document.getElementById('videoUrl');
    const fetchBtn = document.getElementById('fetchBtn');
    const loadingSpinner = document.getElementById('loadingSpinner');
    const errorAlert = document.getElementById('errorAlert');
    const errorMessage = document.getElementById('errorMessage');
    const videoInfo = document.getElementById('videoInfo');
    const playlistInfo = document.getElementById('playlistInfo');
    const downloadProgress = document.getElementById('downloadProgress');
    const downloadStatus = document.getElementById('downloadStatus');
    const batchProgress = document.getElementById('batchProgress');
    
    // Video info elements
    const videoThumbnail = document.getElementById('videoThumbnail');
    const videoTitle = document.getElementById('videoTitle');
    const videoDuration = document.getElementById('videoDuration');
    const qualitySelect = document.getElementById('qualitySelect');
    const audioFormat = document.getElementById('audioFormat');
    const downloadVideoBtn = document.getElementById('downloadVideoBtn');
    const downloadAudioBtn = document.getElementById('downloadAudioBtn');

    // Playlist elements
    const playlistTitle = document.getElementById('playlistTitle');
    const playlistCount = document.getElementById('playlistCount');
    const playlistVideos = document.getElementById('playlistVideos');
    const playlistDownloadType = document.getElementById('playlistDownloadType');
    const playlistAudioFormat = document.getElementById('playlistAudioFormat');
    const downloadPlaylistBtn = document.getElementById('downloadPlaylistBtn');

    // Batch progress elements
    const batchProgressBar = document.getElementById('batchProgressBar');
    const batchProgressText = document.getElementById('batchProgressText');
    const batchStatus = document.getElementById('batchStatus');
    const batchCurrent = document.getElementById('batchCurrent');
    const batchProgressCount = document.getElementById('batchProgressCount');
    const batchDownloadReady = document.getElementById('batchDownloadReady');
    const downloadBatchBtn = document.getElementById('downloadBatchBtn');

    let currentVideoData = null;
    let currentBatchId = null;

    // Utility functions
    function showElement(element) {
        element.style.display = '';
        element.classList.add('fade-in');
    }

    function hideElement(element) {
        element.style.display = 'none';
        element.classList.remove('fade-in');
    }

    function showError(message) {
        errorMessage.textContent = message;
        showElement(errorAlert);
        hideElement(loadingSpinner);
        hideElement(videoInfo);
        hideElement(playlistInfo);
        hideElement(downloadProgress);
        hideElement(batchProgress);
    }

    function hideError() {
        hideElement(errorAlert);
    }

    function formatDuration(seconds) {
        const hours = Math.floor(seconds / 3600);
        const minutes = Math.floor((seconds % 3600) / 60);
        const secs = seconds % 60;
        
        if (hours > 0) {
            return `${hours}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
        } else {
            return `${minutes}:${secs.toString().padStart(2, '0')}`;
        }
    }

    function formatFileSize(bytes) {
        if (!bytes) return 'Unknown size';
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(1024));
        return Math.round(bytes / Math.pow(1024, i) * 100) / 100 + ' ' + sizes[i];
    }

    // Form submission handler
    urlForm.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        const url = videoUrlInput.value.trim();
        if (!url) {
            showError('Please enter a YouTube URL');
            return;
        }

        // Hide previous results and show loading
        hideError();
        hideElement(videoInfo);
        hideElement(playlistInfo);
        hideElement(downloadProgress);
        hideElement(batchProgress);
        showElement(loadingSpinner);
        
        fetchBtn.disabled = true;
        fetchBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Loading...';

        try {
            const formData = new FormData();
            formData.append('url', url);

            const response = await fetch('/get_video_info', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || 'Failed to fetch video information');
            }

            // Store data
            currentVideoData = data;
            
            if (data.type === 'playlist') {
                // Handle playlist
                playlistTitle.textContent = data.title;
                playlistCount.textContent = `${data.total_videos} videos`;
                
                // Populate playlist videos
                playlistVideos.innerHTML = '';
                data.videos.forEach((video, index) => {
                    const videoCard = document.createElement('div');
                    videoCard.className = 'col-md-6 col-lg-4 mb-3';
                    videoCard.innerHTML = `
                        <div class="playlist-video-card slide-in">
                            <img src="${video.thumbnail}" alt="${video.title}" class="playlist-thumbnail mb-2">
                            <h6 class="text-truncate">${video.title}</h6>
                            <small class="text-muted">${formatDuration(video.length)}</small>
                        </div>
                    `;
                    playlistVideos.appendChild(videoCard);
                });
                
                hideElement(loadingSpinner);
                showElement(playlistInfo);
            } else {
                // Handle single video
                videoThumbnail.src = data.thumbnail;
                videoTitle.textContent = data.title;
                videoDuration.textContent = formatDuration(data.length);

                // Populate quality options
                qualitySelect.innerHTML = '<option value="">Choose quality...</option>';
                data.qualities.forEach(quality => {
                    const option = document.createElement('option');
                    option.value = quality.itag;
                    option.textContent = `${quality.quality} (${formatFileSize(quality.filesize)})`;
                    qualitySelect.appendChild(option);
                });

                // Enable audio download if available
                downloadAudioBtn.disabled = !data.has_audio;
                
                hideElement(loadingSpinner);
                showElement(videoInfo);
            }

        } catch (error) {
            console.error('Error fetching video info:', error);
            showError(error.message || 'Failed to fetch video information');
        } finally {
            fetchBtn.disabled = false;
            fetchBtn.innerHTML = '<i class="fas fa-search me-2"></i>Get Video Info';
        }
    });

    // Quality selection handler
    qualitySelect.addEventListener('change', function() {
        downloadVideoBtn.disabled = !this.value;
    });

    // Video download handler
    downloadVideoBtn.addEventListener('click', async function() {
        const selectedQuality = qualitySelect.value;
        if (!selectedQuality || !currentVideoData) {
            showError('Please select a quality first');
            return;
        }

        await downloadFile('/download_video', {
            url: videoUrlInput.value.trim(),
            itag: selectedQuality
        }, 'video');
    });

    // Audio download handler
    downloadAudioBtn.addEventListener('click', async function() {
        if (!currentVideoData) {
            showError('Please fetch video information first');
            return;
        }

        await downloadFile('/download_audio', {
            url: videoUrlInput.value.trim(),
            format: audioFormat.value
        }, 'audio');
    });

    // Generic download function
    async function downloadFile(endpoint, data, type) {
        try {
            // Show download progress
            hideError();
            showElement(downloadProgress);
            downloadStatus.textContent = `Preparing ${type} download...`;

            // Disable download buttons
            downloadVideoBtn.disabled = true;
            downloadAudioBtn.disabled = true;

            const formData = new FormData();
            Object.keys(data).forEach(key => {
                formData.append(key, data[key]);
            });

            const response = await fetch(endpoint, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || `Failed to download ${type}`);
            }

            // Update status
            downloadStatus.textContent = `Processing ${type}...`;

            // Create blob and download
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.style.display = 'none';
            a.href = url;
            
            // Get filename from response headers
            const contentDisposition = response.headers.get('Content-Disposition');
            let filename = `download.${type === 'audio' ? 'mp3' : 'mp4'}`;
            if (contentDisposition) {
                const matches = contentDisposition.match(/filename[^;=\n]*=((['"]).*?\2|[^;\n]*)/);
                if (matches != null && matches[1]) {
                    filename = matches[1].replace(/['"]/g, '');
                }
            }
            
            a.download = filename;
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            document.body.removeChild(a);

            // Hide progress and show success
            hideElement(downloadProgress);
            
            // Show success message
            const successAlert = document.createElement('div');
            successAlert.className = 'alert alert-success fade-in';
            successAlert.innerHTML = `
                <i class="fas fa-check-circle me-2"></i>
                ${type.charAt(0).toUpperCase() + type.slice(1)} downloaded successfully!
            `;
            errorAlert.parentNode.insertBefore(successAlert, errorAlert);
            
            // Remove success message after 5 seconds
            setTimeout(() => {
                if (successAlert.parentNode) {
                    successAlert.parentNode.removeChild(successAlert);
                }
            }, 5000);

        } catch (error) {
            console.error(`Error downloading ${type}:`, error);
            hideElement(downloadProgress);
            showError(error.message || `Failed to download ${type}`);
        } finally {
            // Re-enable download buttons
            downloadVideoBtn.disabled = !qualitySelect.value;
            downloadAudioBtn.disabled = !currentVideoData?.has_audio;
        }
    }

    // Playlist download handler
    downloadPlaylistBtn.addEventListener('click', async function() {
        if (!currentVideoData || currentVideoData.type !== 'playlist') {
            showError('Please fetch playlist information first');
            return;
        }

        try {
            hideError();
            showElement(batchProgress);
            
            const formData = new FormData();
            formData.append('url', videoUrlInput.value.trim());
            formData.append('type', playlistDownloadType.value);
            formData.append('format', playlistAudioFormat.value);

            const response = await fetch('/download_playlist', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || 'Failed to start playlist download');
            }

            currentBatchId = data.batch_id;
            trackBatchProgress();

        } catch (error) {
            console.error('Error starting playlist download:', error);
            hideElement(batchProgress);
            showError(error.message || 'Failed to start playlist download');
        }
    });

    // Batch download handler
    downloadBatchBtn.addEventListener('click', async function() {
        if (!currentBatchId) {
            showError('No batch download available');
            return;
        }

        try {
            const response = await fetch(`/download_batch/${currentBatchId}`);
            
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || 'Failed to download batch');
            }

            // Create blob and download
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.style.display = 'none';
            a.href = url;
            a.download = `youtube_batch_${currentBatchId}.zip`;
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            document.body.removeChild(a);

            // Reset batch state
            currentBatchId = null;
            hideElement(batchProgress);

        } catch (error) {
            console.error('Error downloading batch:', error);
            showError(error.message || 'Failed to download batch');
        }
    });

    // Track batch progress
    function trackBatchProgress() {
        if (!currentBatchId) return;

        const interval = setInterval(async () => {
            try {
                const response = await fetch(`/batch_progress/${currentBatchId}`);
                const progress = await response.json();

                if (!response.ok) {
                    clearInterval(interval);
                    showError('Failed to track progress');
                    return;
                }

                updateBatchProgress(progress);

                if (progress.status === 'completed' || progress.status === 'error') {
                    clearInterval(interval);
                }

            } catch (error) {
                console.error('Error tracking progress:', error);
                clearInterval(interval);
            }
        }, 2000);
    }

    // Update batch progress display
    function updateBatchProgress(progress) {
        const percentage = progress.total > 0 ? Math.round((progress.current / progress.total) * 100) : 0;
        
        batchProgressBar.style.width = `${percentage}%`;
        batchProgressText.textContent = `${percentage}%`;
        batchStatus.textContent = getStatusText(progress.status);
        batchCurrent.textContent = progress.current_video || '-';
        batchProgressCount.textContent = `${progress.current} / ${progress.total}`;

        if (progress.status === 'completed') {
            batchProgressBar.classList.remove('progress-bar-animated');
            batchProgressBar.classList.add('bg-success');
            showElement(batchDownloadReady);
        } else if (progress.status === 'error') {
            batchProgressBar.classList.remove('progress-bar-animated');
            batchProgressBar.classList.add('bg-danger');
            batchStatus.textContent = `Error: ${progress.error || 'Unknown error'}`;
        }
    }

    // Get human readable status text
    function getStatusText(status) {
        const statusMap = {
            'starting': 'Starting download...',
            'downloading': 'Downloading videos...',
            'creating_zip': 'Creating ZIP file...',
            'completed': 'Download completed!',
            'error': 'Download failed'
        };
        return statusMap[status] || status;
    }

    // Show/hide audio format based on download type
    playlistDownloadType.addEventListener('change', function() {
        const audioFormatSection = document.getElementById('playlistAudioFormatSection');
        if (this.value === 'audio') {
            showElement(audioFormatSection);
        } else {
            hideElement(audioFormatSection);
        }
    });

    // Auto-focus URL input
    videoUrlInput.focus();

    // Clear error when user starts typing
    videoUrlInput.addEventListener('input', function() {
        if (errorAlert.style.display !== 'none') {
            hideError();
        }
    });
});