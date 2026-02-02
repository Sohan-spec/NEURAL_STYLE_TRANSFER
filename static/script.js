document.addEventListener('DOMContentLoaded', () => {
    const styleSlider = document.getElementById('style-slider');
    const fileInput = document.getElementById('file-input');
    const dropZone = document.getElementById('drop-zone');
    const previewImg = document.getElementById('preview-img');
    const uploadPlaceholder = document.getElementById('upload-placeholder');
    const transformBtn = document.getElementById('transform-btn');
    const bgLayer = document.getElementById('bg-layer');
    const progressModal = document.getElementById('progress-modal');
    const progressBar = document.getElementById('progress-bar');
    const progressText = document.getElementById('progress-text');
    const logText = document.getElementById('log-text');
    const resultOverlay = document.getElementById('result-overlay');
    const resultImg = document.getElementById('result-img');
    const downloadBtn = document.getElementById('download-btn');
    const previewWrapper = document.getElementById('preview-wrapper');
    const removeContentBtn = document.getElementById('remove-content-btn');
    const customStyleInput = document.getElementById('custom-style-input');

    let selectedStyle = '';
    let selectedFile = null;

    // Load Styles
    fetch('/styles')
        .then(response => response.json())
        .then(styles => {
            styles.forEach(style => {
                const img = document.createElement('img');
                img.src = `/data/style-images/${style}`;
                img.classList.add('style-item');
                img.onclick = () => selectStyle(style, img);
                styleSlider.appendChild(img);
            });
            // Auto-select Starry Night or first
            const defaultStyle = 'vg_starry_night.jpg';
            if (styles.includes(defaultStyle)) {
                // We need to find the img element that corresponds to this style
                // Since we just appended them, we can iterate or finding by src
                const images = Array.from(styleSlider.getElementsByTagName('img'));
                const targetImg = images.find(img => img.src.includes(defaultStyle));
                if (targetImg) selectStyle(defaultStyle, targetImg);
            } else if (styles.length > 0) {
                selectStyle(styles[0], styleSlider.firstChild);
            }

            // Add custom style upload button at the end
            const addBtn = document.createElement('div');
            addBtn.classList.add('add-style-btn');
            addBtn.innerHTML = `
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <line x1="12" y1="5" x2="12" y2="19"></line>
                    <line x1="5" y1="12" x2="19" y2="12"></line>
                </svg>
                <span>Add Style</span>
            `;
            addBtn.onclick = () => customStyleInput.click();
            styleSlider.appendChild(addBtn);
        });

    const styleToast = document.getElementById('style-toast');
    let toastTimeout = null;

    // Format style name: "vg_starry_night.jpg" -> "Starry Night"
    function formatStyleName(filename) {
        return filename
            .replace(/\.(jpg|jpeg|png|bmp)$/i, '') // Remove extension
            .replace(/^vg_/, '')                   // Remove vg_ prefix
            .replace(/^temp_[a-f0-9]+_/, '')       // Remove temp_{uuid}_ prefix
            .replace(/^custom_style_/, '')         // Remove custom style prefix  
            .replace(/_/g, ' ')                    // Replace underscores with spaces
            .split(' ')
            .map(word => word.charAt(0).toUpperCase() + word.slice(1)) // Capitalize each word
            .join(' ');
    }

    function showStyleToast(styleName) {
        // Clear any existing timeout to prevent overlap
        if (toastTimeout) {
            clearTimeout(toastTimeout);
        }

        const formattedName = formatStyleName(styleName);
        styleToast.textContent = formattedName;
        styleToast.classList.add('show');

        // Hide after 1.5 seconds
        toastTimeout = setTimeout(() => {
            styleToast.classList.remove('show');
        }, 1500);
    }

    function selectStyle(styleName, element) {
        selectedStyle = styleName;
        // Update UI
        document.querySelectorAll('.style-item').forEach(el => el.classList.remove('selected'));
        element.classList.add('selected');

        // Update Background - use correct path for temp vs regular styles
        const bgUrl = styleName.startsWith('temp_')
            ? `/data/temp-styles/${styleName}`
            : `/data/style-images/${styleName}`;
        bgLayer.style.backgroundImage = `url(${bgUrl})`;

        // Show toast
        showStyleToast(styleName);

        checkReady();
    }

    // File Upload
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('dragover');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        if (e.dataTransfer.files.length > 0) {
            handleFile(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFile(e.target.files[0]);
        }
    });

    function handleFile(file) {
        if (!file.type.startsWith('image/')) {
            alert('Please upload an image file');
            return;
        }
        selectedFile = file;
        const reader = new FileReader();
        reader.onload = (e) => {
            previewImg.src = e.target.result;
            previewWrapper.hidden = false;
            uploadPlaceholder.hidden = true;
        }
        reader.readAsDataURL(file);

        // Upload to server immediately to get filename/prep
        const formData = new FormData();
        formData.append('image', file);

        fetch('/upload_content', {
            method: 'POST',
            body: formData
        })
            .then(res => res.json())
            .then(data => {
                if (data.filename) {
                    console.log('Uploaded:', data.filename);
                }
            });

        checkReady();
    }

    function checkReady() {
        if (selectedStyle && selectedFile) {
            transformBtn.disabled = false;
        } else {
            transformBtn.disabled = true;
        }
    }

    // Remove content image
    removeContentBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        selectedFile = null;
        previewImg.src = '';
        previewWrapper.hidden = true;
        uploadPlaceholder.hidden = false;
        fileInput.value = '';
        checkReady();
    });

    // Custom style upload
    customStyleInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            const file = e.target.files[0];
            if (!file.type.startsWith('image/')) {
                alert('Please upload an image file');
                return;
            }

            const formData = new FormData();
            formData.append('image', file);

            fetch('/upload_style', {
                method: 'POST',
                body: formData
            })
                .then(res => res.json())
                .then(data => {
                    if (data.filename && data.url) {
                        // Create new style item
                        const img = document.createElement('img');
                        img.src = data.url;
                        img.classList.add('style-item');
                        img.onclick = () => selectStyle(data.filename, img);

                        // Insert before the "Add Style" button
                        const addBtn = styleSlider.querySelector('.add-style-btn');
                        styleSlider.insertBefore(img, addBtn);

                        // Auto-select the new style
                        selectStyle(data.filename, img);
                    }
                });
        }
    });

    // Transform
    transformBtn.addEventListener('click', () => {
        if (!selectedFile || !selectedStyle) return;

        progressModal.classList.remove('hidden');
        progressBar.style.width = '0%';
        progressText.innerText = '0% - Initializing...';

        const contentName = selectedFile.name; // Assuming upload handled preserves name or we should rely on response
        // Note: For this demo we assume the upload preserves the filename in the content folder.
        // In real app, we should use the one returned by /upload_content.

        const eventSource = new EventSource(`/predict?content_img_name=${contentName}&style_img_name=${selectedStyle}`);

        let startTime = Date.now();

        eventSource.onmessage = (event) => {
            const data = JSON.parse(event.data);

            if (data.progress) {
                progressBar.style.width = `${data.progress}%`;

                // Estimate time
                const elapsed = (Date.now() - startTime) / 1000;
                const percent = data.progress;
                if (percent > 0) {
                    const totalEstimated = elapsed / (percent / 100);
                    const remaining = Math.max(0, totalEstimated - elapsed).toFixed(0);
                    progressText.innerText = `${percent}% - approx ${remaining}s left`;
                }
            }

            if (data.log) {
                logText.innerText = data.log;
            }

            if (data.status === 'done') {
                eventSource.close();
                progressModal.classList.add('hidden');
                resultImg.src = data.url;

                // Ensure download is JPG with proper filename
                const styleName = selectedStyle.replace('.jpg', '').replace('.png', '');
                const contentName = selectedFile.name.split('.')[0];
                downloadBtn.href = data.url;
                downloadBtn.download = `${contentName}_${styleName}_artflow.jpg`;

                resultOverlay.classList.remove('hidden');
            }

            if (data.error) {
                eventSource.close();
                alert('Error: ' + data.error);
                progressModal.classList.add('hidden');
            }
        };

        eventSource.onerror = () => {
            eventSource.close();
            // alert('Connection lost');
            // progressModal.classList.add('hidden');
        };
    });

    window.closeResult = () => {
        resultOverlay.classList.add('hidden');
    };
});
