// Utility functions module
export function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

export function validateProjectPath(path) {
    return path.trim().length > 0;
}

export function setButtonState(button, isLoading) {
    button.disabled = isLoading;
    button.classList.toggle('loading', isLoading);
}