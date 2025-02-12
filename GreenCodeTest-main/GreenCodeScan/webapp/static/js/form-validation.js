// Form validation module
import { showAlert } from './alerts.js';
import { handleFormSubmission } from './form-submission.js';

export function validateForm() {
    const projectPath = document.getElementById('project_path').value.trim();
    const promptSelect = document.getElementById('prompt_1');
    let isPromptSelected = false;

    if (!projectPath) {
        showAlert("The 'Project Path' field is mandatory.", false);
        return false;
    }

    // Check if the unified prompt is enabled
    if (promptSelect.value === "y") {
        isPromptSelected = true;
    } else {
        promptSelect.classList.add('error');
    }

    if (!isPromptSelected) {
        showAlert("The unified 'Configure Prompt' must be set to 'Yes'.", false);
        return false;
    }

    return true;
}

// Initialize form validation
document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('config-form');
    form.addEventListener('submit', handleFormSubmission);
});