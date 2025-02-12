// Form submission module
import { showAlert } from './alerts.js';
import { validateForm } from './form-validation.js';

export async function handleFormSubmission(event) {
    event.preventDefault();

    if (!validateForm()) return;

    const form = document.getElementById('config-form');
    const saveButton = document.getElementById('save-config');
    const runButton = document.getElementById('run-button');
    const inputs = form.querySelectorAll('input, select');

    // Add loading state
    saveButton.classList.add('loading');
    saveButton.disabled = true;

    try {
        const formData = new FormData(form);
        const response = await fetch(form.action, {
            method: form.method,
            body: formData
        });

        if (response.ok) {
            showAlert("Configuration saved! You can now start Green CodeRefiner!", true);
            inputs.forEach(input => input.disabled = true);
            runButton.disabled = false;
        } else {
            throw new Error('Failed to save configuration');
        }
    } catch (error) {
        console.error('Error:', error);
        showAlert("An error occurred while saving the configuration. Please try again.", false);
        saveButton.disabled = false;
    } finally {
        // Remove loading state and change button text
        setTimeout(() => {
            saveButton.classList.remove('loading');
            saveButton.querySelector('.button-text').innerHTML = `
                <i class="fas fa-check"></i>
                Saved
            `;
        }, 1000);
    }
}