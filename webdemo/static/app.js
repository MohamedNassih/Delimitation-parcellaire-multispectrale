// Agricultural Boundary Detection Demo JavaScript

class BoundaryDetector {
    constructor() {
        this.apiUrl = '';
        this.initEventListeners();
        this.checkHealth();
    }

    initEventListeners() {
        const fileInput = document.getElementById('file-upload');
        const uploadBtn = document.getElementById('upload-btn');
        const uploadArea = document.getElementById('upload-area');
        const retryBtn = document.getElementById('retry-btn');
        const downloadBtn = document.getElementById('download-btn');

        // File input change
        fileInput.addEventListener('change', (e) => {
            const file = e.target.files[0];
            if (file) {
                this.validateFile(file);
                this.previewFile(file);
                uploadBtn.disabled = false;
                uploadBtn.textContent = `Analyze ${file.name}`;
            }
        });

        // Upload button click
        uploadBtn.addEventListener('click', () => {
            const file = fileInput.files[0];
            if (file) {
                this.uploadAndAnalyze(file);
            }
        });

        // Drag and drop functionality
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('border-primary', 'bg-green-50');
        });

        uploadArea.addEventListener('dragleave', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('border-primary', 'bg-green-50');
        });

        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('border-primary', 'bg-green-50');
            
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                const file = files[0];
                fileInput.files = files;
                this.validateFile(file);
                this.previewFile(file);
                uploadBtn.disabled = false;
                uploadBtn.textContent = `Analyze ${file.name}`;
            }
        });

        // Retry button
        retryBtn.addEventListener('click', () => {
            this.resetUI();
        });

        // Download button
        downloadBtn.addEventListener('click', () => {
            this.downloadResult();
        });

        // Update status indicator
        uploadArea.addEventListener('click', () => {
            fileInput.click();
        });
    }

    async checkHealth() {
        try {
            const response = await fetch('/health');
            const data = await response.json();
            
            const indicator = document.getElementById('status-indicator');
            if (data.status === 'healthy') {
                indicator.classList.remove('bg-red-400');
                indicator.classList.add('bg-green-400');
            } else {
                indicator.classList.remove('bg-green-400');
                indicator.classList.add('bg-red-400');
            }
        } catch (error) {
            console.error('Health check failed:', error);
            const indicator = document.getElementById('status-indicator');
            indicator.classList.remove('bg-green-400');
            indicator.classList.add('bg-red-400');
        }
    }

    validateFile(file) {
        const allowedTypes = ['image/png', 'image/jpeg', 'image/jpg', 'image/tiff', 'image/x-tiff'];
        const allowedExtensions = ['.png', '.jpg', '.jpeg', '.tif', '.tiff'];
        const maxSize = 50 * 1024 * 1024; // 50MB

        if (!allowedTypes.includes(file.type) && !allowedExtensions.some(ext => file.name.toLowerCase().endsWith(ext))) {
            throw new Error('Unsupported file type. Please upload PNG, JPG, or GeoTIFF files.');
        }

        if (file.size > maxSize) {
            throw new Error('File too large. Maximum size is 50MB.');
        }
    }

    previewFile(file) {
        const reader = new FileReader();
        reader.onload = (e) => {
            const img = document.getElementById('original-image');
            img.src = e.target.result;
        };
        reader.readAsDataURL(file);
    }

    showLoading() {
        this.hideAllSections();
        document.getElementById('loading-state').classList.remove('hidden');
        
        const messages = [
            'Loading AI model...',
            'Preprocessing image...',
            'Analyzing multispectral data...',
            'Detecting field boundaries...',
            'Generating results...'
        ];
        
        let messageIndex = 0;
        const messageElement = document.getElementById('loading-message');
        
        const updateMessage = () => {
            if (messageIndex < messages.length) {
                messageElement.textContent = messages[messageIndex];
                messageIndex++;
                setTimeout(updateMessage, 1000);
            }
        };
        
        updateMessage();
    }

    hideLoading() {
        document.getElementById('loading-state').classList.add('hidden');
    }

    showResults(data) {
        this.hideAllSections();
        
        // Update result image
        const resultImg = document.getElementById('result-image');
        resultImg.src = `data:image/png;base64,${data.result.mask_base64_png}`;
        
        // Update metrics
        document.getElementById('inference-time').textContent = `${data.inference_time_seconds}s`;
        document.getElementById('image-dimensions').textContent = `${data.result.mask_dimensions.width}×${data.result.mask_dimensions.height}`;
        document.getElementById('boundary-ratio').textContent = `${(data.result.mask_stats.positive_ratio * 100).toFixed(1)}%`;
        
        // Update pipeline info
        document.getElementById('pipeline-type').textContent = this.getPipelineDescription(data.pipeline_type);
        
        // Update image info
        const imageInfo = data.image_info;
        document.getElementById('image-info').textContent = 
            `${imageInfo.filename} • ${imageInfo.width}×${imageInfo.height} • ${imageInfo.format || imageInfo.bands} bands`;
        
        // Show results section
        document.getElementById('results-section').classList.remove('hidden');
    }

    showError(message) {
        this.hideAllSections();
        document.getElementById('error-message').textContent = message;
        document.getElementById('error-state').classList.remove('hidden');
    }

    hideAllSections() {
        document.getElementById('loading-state').classList.add('hidden');
        document.getElementById('results-section').classList.add('hidden');
        document.getElementById('error-state').classList.add('hidden');
    }

    resetUI() {
        document.getElementById('file-upload').value = '';
        document.getElementById('upload-btn').disabled = true;
        document.getElementById('upload-btn').textContent = 'Choose File to Analyze';
        this.hideAllSections();
        
        // Reset preview
        const originalImg = document.getElementById('original-image');
        const resultImg = document.getElementById('result-image');
        originalImg.src = '';
        resultImg.src = '';
    }

    getPipelineDescription(pipelineType) {
        const descriptions = {
            'png_jpg_lightweight': 'Lightweight pipeline for PNG/JPG images. Optimized for quick processing.',
            'geotiff_full': 'Full pipeline for GeoTIFF images. Uses all available spectral bands for enhanced accuracy.'
        };
        return descriptions[pipelineType] || 'Unknown pipeline';
    }

    async uploadAndAnalyze(file) {
        this.showLoading();
        
        try {
            const formData = new FormData();
            formData.append('file', file);
            
            const response = await fetch('/infer', {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || `HTTP ${response.status}: ${response.statusText}`);
            }
            
            const result = await response.json();
            this.showResults(result);
            
        } catch (error) {
            console.error('Analysis failed:', error);
            this.showError(error.message || 'Failed to analyze image. Please try again.');
        } finally {
            this.hideLoading();
        }
    }

    downloadResult() {
        const resultImg = document.getElementById('result-image');
        if (resultImg.src) {
            const link = document.createElement('a');
            link.download = `boundary_detection_${Date.now()}.png`;
            link.href = resultImg.src;
            link.click();
        }
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new BoundaryDetector();
});

// Export for potential testing
if (typeof module !== 'undefined' && module.exports) {
    module.exports = BoundaryDetector;
}