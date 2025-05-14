document.addEventListener('DOMContentLoaded', function() {
    // Initialize chart with empty data
    let pollutionChart = null;
    
    // Handle model upload form submission
    const modelUploadForm = document.getElementById('model-upload-form');
    if (modelUploadForm) {
        modelUploadForm.addEventListener('submit', function(e) {
            e.preventDefault();
            
            const formData = new FormData(modelUploadForm);
            const uploadMessage = document.getElementById('upload-message');
            
            fetch('/upload_model', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    uploadMessage.innerHTML = `<p class="error-message">${data.error}</p>`;
                } else {
                    uploadMessage.innerHTML = `<p class="success-message">${data.message}</p>`;
                    // Enable the prediction button
                    document.querySelector('#prediction-form button').disabled = false;
                    // Hide the upload form after a delay
                    setTimeout(() => {
                        document.querySelector('.model-upload').style.display = 'none';
                    }, 3000);
                }
            })
            .catch(error => {
                uploadMessage.innerHTML = `<p class="error-message">Error: ${error.message}</p>`;
            });
        });
    }
    
    // Handle prediction form submission
    const predictionForm = document.getElementById('prediction-form');
    predictionForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        // Collect form data
        const formData = new FormData(predictionForm);
        const formDataObj = {};
        for (let [key, value] of formData.entries()) {
            formDataObj[key] = value;
        }
        
        // Make prediction request
        fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(formDataObj)
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                alert(`Error: ${data.error}`);
            } else {
                displayResults(data);
            }
        })
        .catch(error => {
            alert(`Error: ${error.message}`);
        });
    });
    
    // Function to display prediction results
    function displayResults(data) {
        // Show results section
        document.getElementById('results-section').style.display = 'block';
        
        // Display prediction value
        const predictionValue = document.getElementById('prediction-value');
        predictionValue.textContent = Array.isArray(data.prediction) ? 
            data.prediction[0].toFixed(2) : 
            data.prediction.toFixed(2);
        
        // Display recommendation based on AQI value
        const recommendationDiv = document.getElementById('recommendation');
        const aqi = parseFloat(data.input_data.AQI);
        let recommendationText, recommendationClass;
        
        if (aqi <= 50) {
            recommendationText = "Air quality is good. Enjoy outdoor activities!";
            recommendationClass = "good";
        } else if (aqi <= 100) {
            recommendationText = "Air quality is acceptable. Unusually sensitive people should consider reducing prolonged outdoor exertion.";
            recommendationClass = "moderate";
        } else if (aqi <= 150) {
            recommendationText = "Members of sensitive groups may experience health effects. The general public is less likely to be affected.";
            recommendationClass = "moderate";
        } else if (aqi <= 200) {
            recommendationText = "Health alert: everyone may experience health effects. Members of sensitive groups may experience more serious health effects.";
            recommendationClass = "poor";
        } else if (aqi <= 300) {
            recommendationText = "Health warnings of emergency conditions. The entire population is more likely to be affected.";
            recommendationClass = "poor";
        } else {
            recommendationText = "Health alert: everyone may experience more serious health effects. Avoid outdoor activities.";
            recommendationClass = "poor";
        }
        
        recommendationDiv.textContent = recommendationText;
        recommendationDiv.className = `recommendation ${recommendationClass}`;
        
        // Update pollution chart
        updatePollutionChart(data.input_data);
    }
    
    // Function to update pollution chart
    function updatePollutionChart(inputData) {
        // Extract pollutant data excluding AQI and AQI_Bucket
        const pollutants = Object.keys(inputData).filter(key => key !== 'AQI' && key !== 'AQI_Bucket');
        const values = pollutants.map(key => inputData[key]);
        
        // Define standard thresholds for common pollutants (simplified for visualization)
        const thresholds = {
            'PM2.5': 25,  // WHO guideline
            'PM10': 50,   // WHO guideline
            'NO2': 40,    // WHO guideline
            'SO2': 20,    // WHO guideline
            'O3': 100,    // WHO guideline
            'CO': 10,     // WHO guideline
            'NO': 25,     // Approximation
            'NOx': 40,    // Approximation
            'NH3': 100,   // Approximation
            'Benzene': 5, // Approximation
            'Toluene': 260, // Approximation
            'Xylene': 100   // Approximation
        };
        
        // Prepare threshold data
        const thresholdData = pollutants.map(key => thresholds[key] || 0);
        
        // Generate colors based on whether the value exceeds threshold
        const backgroundColors = values.map((value, index) => {
            const threshold = thresholdData[index];
            return value > threshold ? 'rgba(255, 99, 132, 0.7)' : 'rgba(75, 192, 192, 0.7)';
        });
        
        // Create or update chart
        if (pollutionChart) {
            pollutionChart.data.labels = pollutants;
            pollutionChart.data.datasets[0].data = values;
            pollutionChart.data.datasets[0].backgroundColor = backgroundColors;
            if (pollutionChart.data.datasets.length > 1) {
                pollutionChart.data.datasets[1].data = thresholdData;
            } else {
                pollutionChart.data.datasets.push({
                    type: 'line',
                    label: 'Threshold',
                    data: thresholdData,
                    borderColor: 'rgba(255, 159, 64, 1)',
                    borderWidth: 2,
                    fill: false,
                    pointRadius: 0
                });
            }
            pollutionChart.update();
        } else {
            const ctx = document.getElementById('pollution-chart').getContext('2d');
            pollutionChart = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: pollutants,
                    datasets: [{
                        label: 'Current Values',
                        data: values,
                        backgroundColor: backgroundColors,
                        borderColor: backgroundColors.map(color => color.replace('0.7', '1')),
                        borderWidth: 1
                    }, {
                        type: 'line',
                        label: 'Threshold',
                        data: thresholdData,
                        borderColor: 'rgba(255, 159, 64, 1)',
                        borderWidth: 2,
                        fill: false,
                        pointRadius: 0
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: true,
                            title: {
                                display: true,
                                text: 'Concentration'
                            }
                        },
                        x: {
                            title: {
                                display: true,
                                text: 'Pollutants'
                            },
                            ticks: {
                                maxRotation: 45,
                                minRotation: 45
                            }
                        }
                    },
                    plugins: {
                        tooltip: {
                            callbacks: {
                                afterLabel: function(context) {
                                    const index = context.dataIndex;
                                    const pollutant = pollutants[index];
                                    const threshold = thresholds[pollutant];
                                    const value = values[index];
                                    
                                    if (value > threshold) {
                                        return `Exceeds threshold by ${(value - threshold).toFixed(2)}`;
                                    } else {
                                        return `Below threshold by ${(threshold - value).toFixed(2)}`;
                                    }
                            }
                        },
                        legend: {
                            display: true,
                            position: 'top'
                        },
                        title: {
                            display: true,
                            text: 'Pollution Parameters vs Thresholds'
                        }
                    }
                }
            });
        }
    }
    
    // Auto-update AQI category based on AQI value selection
    const aqiInput = document.getElementById('AQI');
    const aqiBucketSelect = document.getElementById('AQI_Bucket');
    
    aqiInput.addEventListener('change', function() {
        const aqiValue = parseFloat(this.value);
        
        if (aqiValue <= 50) {
            aqiBucketSelect.value = 'Good';
        } else if (aqiValue <= 100) {
            aqiBucketSelect.value = 'Satisfactory';
        } else if (aqiValue <= 200) {
            aqiBucketSelect.value = 'Moderate';
        } else if (aqiValue <= 300) {
            aqiBucketSelect.value = 'Poor';
        } else if (aqiValue <= 400) {
            aqiBucketSelect.value = 'Very Poor';
        } else {
            aqiBucketSelect.value = 'Severe';
        }
    });
});
                }