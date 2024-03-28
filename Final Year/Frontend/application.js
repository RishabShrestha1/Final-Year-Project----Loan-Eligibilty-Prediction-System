// Get form and modal elements
const predictionForm = document.getElementById('prediction-form');
const predictionModal = document.getElementById('prediction-modal');
const predictionResult = document.getElementById('prediction-result');
const closeButton = document.getElementsByClassName('close-button')[0];

// Handle form submission
predictionForm.addEventListener('submit', async (event) => {
    event.preventDefault();

    // Get form data
    const formData = new FormData(event.target);
    const dependents = formData.get('dependents');
    const education = formData.get('education');
    const selfEmployed = formData.get('self-employed');
    const annualIncome = formData.get('annual-income');
    const loanTerm = formData.get('loan-term');
    const cibilScore = formData.get('cibil-score');
    const residentialAssets = formData.get('residential-assets');
    const commercialAssets = formData.get('commercial-assets');
    const luxuryAssets = formData.get('luxury-assets');
    const bankAssets = formData.get('bank-assets');

    try {
        // Send data to Gradio API
        const app = await client("http://127.0.0.1:7860/");
        const result = await app.predict("/predict", [
            dependents,
            education,
            selfEmployed,
            annualIncome,
            loanTerm,
            cibilScore,
            residentialAssets,
            commercialAssets,
            luxuryAssets,
            bankAssets
        ]);

        // Display prediction in modal
        predictionResult.textContent = result.data;
        predictionModal.style.display = 'block';
    } catch (error) {
        console.error('Error:', error);
    }
});

// Close modal when clicked outside
window.onclick = function(event) {
    if (event.target === predictionModal) {
        predictionModal.style.display = 'none';
    }
}

// Close modal when close button is clicked
closeButton.onclick = function() {
    predictionModal.style.display = 'none';
}