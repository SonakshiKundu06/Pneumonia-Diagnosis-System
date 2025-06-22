document.addEventListener('DOMContentLoaded', function() {
    // Navigation handling
    const navLinks = document.querySelectorAll('.main-nav a');
    const sections = document.querySelectorAll('.section');
    const fileInput = document.getElementById('fileInput');
    const fileName = document.getElementById('fileName');
    const resultSection = document.getElementById('result');
    const imagePreview = document.getElementById('imagePreview');
    const predictionElement = document.getElementById('prediction');
    const confidenceElement = document.getElementById('confidence');
  
    // Report elements
    const reportDate = document.getElementById('reportDate');
    const reportPrediction = document.getElementById('reportPrediction');
    const reportConfidence = document.getElementById('reportConfidence');
  
    // Store prediction results
    let predictionData = {
      date: null,
      prediction: null,
      confidence: null,
      imageData: null
    };
  
    // Navigation functionality
    function navigateTo(sectionId) {
      sections.forEach(section => {
        section.classList.remove('active');
      });
      
      navLinks.forEach(link => {
        link.classList.remove('active');
        if (link.getAttribute('href') === '#' + sectionId) {
          link.classList.add('active');
        }
      });
      
      const targetSection = document.getElementById(sectionId);
      if (targetSection) {
        targetSection.classList.add('active');
      }
    }
  
    // Handle navigation clicks
    navLinks.forEach(link => {
      link.addEventListener('click', function(e) {
        e.preventDefault();
        const sectionId = this.getAttribute('href').substring(1);
        navigateTo(sectionId);
      });
    });
  
    // Handle CTA buttons
    document.querySelectorAll('.cta-button').forEach(button => {
      button.addEventListener('click', function(e) {
        e.preventDefault();
        const sectionId = this.getAttribute('href').substring(1);
        navigateTo(sectionId);
      });
    });
  
    // File input handling
    fileInput.addEventListener('change', function() {
      if (this.files && this.files[0]) {
        fileName.textContent = this.files[0].name;
        
        // Create preview
        const reader = new FileReader();
        reader.onload = function(e) {
          imagePreview.src = e.target.result;
        };
        reader.readAsDataURL(this.files[0]);
      } else {
        fileName.textContent = 'No file selected';
        imagePreview.src = '';
      }
      
      // Hide result section when new file is selected
      resultSection.style.display = 'none';
    });
  
    // Global predict function
    window.predict = function() {
      if (!fileInput.files || !fileInput.files[0]) {
        alert('Please select an X-ray image first.');
        return;
      }
  
      // Show loading state
      predictionElement.textContent = 'Analyzing...';
      resultSection.style.display = 'block';
      
      // In a real app, this would send the image to your backend for prediction
      // For demo purposes, we'll simulate a prediction
      callPredictionApi(fileInput.files[0]);
    };
  
    // // Simulate prediction (in production, this would call your backend API)
    // function simulatePrediction(imageFile) {
    //   // Read the file to get image data for our report
    //   const reader = new FileReader();
    //   reader.onload = function(e) {
    //     // Store the image data for reporting
    //     predictionData.imageData = e.target.result;
        
    //     // Simulate API call delay
    //     setTimeout(() => {
    //       // Generate random prediction for demo
    //       // In production, you would call your FastAPI backend instead
    //       const isPneumonia = Math.random() > 0.5;
    //       const prediction = isPneumonia ? 'PNEUMONIA' : 'NORMAL';
    //       const confidence = (60 + Math.random() * 35).toFixed(2); // 60-95% confidence
          
    //       // Update UI
    //       displayPredictionResult(prediction, confidence);
          
    //       // Store data for report
    //       predictionData.date = new Date();
    //       predictionData.prediction = prediction;
    //       predictionData.confidence = confidence;
          
    //       // Update report section
    //       updateReportSection();
    //     }, 1500);
    //   };
    //   reader.readAsDataURL(imageFile);
    // }
  
    // In production, this function would call your backend API
    function callPredictionApi(imageFile) {
      const formData = new FormData();
      formData.append('image', imageFile);
  
      fetch('http://localhost:8000/predict', {
        method: 'POST',
        body: formData
      })
      .then(response => response.json())
      .then(data => {
        displayPredictionResult(data.predicted_class, data.confidence);
        
        // Store data for report
        predictionData.date = new Date();
        predictionData.prediction = data.predicted_class;
        predictionData.confidence = data.confidence;
        predictionData.imageData = `data:image/jpeg;base64,${data.image_base64}`;
        
        // Update report section
        updateReportSection();
      })
      .catch(error => {
        console.error('Error:', error);
        alert('Error processing your image. Please try again.');
        resultSection.style.display = 'none';
      });
    }
  
    function displayPredictionResult(prediction, confidence) {
      const color = prediction === 'NORMAL' ? 'green' : 'red';
      
      predictionElement.textContent = `Prediction: ${prediction}`;
      predictionElement.style.color = color;
      
      confidenceElement.textContent = `Confidence: ${confidence}%`;
      resultSection.style.display = 'block';
    }
  
    function updateReportSection() {
      if (predictionData.date) {
        reportDate.textContent = predictionData.date.toLocaleString();
        reportPrediction.textContent = predictionData.prediction;
        reportConfidence.textContent = `${predictionData.confidence}%`;
        
        // Style based on prediction
        reportPrediction.style.color = predictionData.prediction === 'NORMAL' ? 'green' : 'red';
      }
    }
  

    // Global function to download PDF report
    window.downloadReport = function() {
      if (!predictionData.prediction) {
        alert('Please make a prediction first.');
        navigateTo('upload');
        return;
      }
  
      // Create PDF with jsPDF
      const { jsPDF } = window.jspdf;
      const doc = new jsPDF();
      
      // Add title
      doc.setFontSize(22);
      doc.text('Pneumonia Detection Report', 105, 20, { align: 'center' });
      
      // Add date
      doc.setFontSize(12);
      doc.text(`Date: ${predictionData.date.toLocaleString()}`, 20, 40);
      
      // Add prediction
      doc.setFontSize(16);
      doc.text(`Prediction: ${predictionData.prediction}`, 20, 50);
      
      // Add confidence
      doc.text(`Confidence: ${predictionData.confidence}%`, 20, 60);
      
      // Add image if available
      if (predictionData.imageData) {
        doc.addImage(predictionData.imageData, 'JPEG', 20, 70, 170, 170);
      }
      
      // Add disclaimer
      doc.setFontSize(10);
      doc.text('DISCLAIMER: This is an AI-assisted prediction for educational purposes only.', 105, 250, { align: 'center' });
      doc.text('Please consult with a healthcare professional for proper diagnosis.', 105, 257, { align: 'center' });
      
      // Save the PDF
      doc.save('pneumonia-detection-report.pdf');
    };
  });
