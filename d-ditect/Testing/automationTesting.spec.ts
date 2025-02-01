import { test, expect } from '@playwright/test';
import { locators } from './locators';

test('Login form validation and redirection', async ({ page }) => {
    // Navigate to the login page
    await page.goto('http://127.0.0.1:5000/login');

    // Locate elements using locators from the locators.ts file
    const usernameInput = page.locator(locators.loginPage.usernameInput);
    const loginButton = page.locator(locators.loginPage.loginButton);

    // Step 1: Verify that the login button is initially disabled
    await expect(loginButton).toBeDisabled();

    // Step 2: Type a valid username and verify the button is enabled
    await usernameInput.fill('Sonal');
    await expect(loginButton).toBeEnabled();

    await page.waitForTimeout(1000);

    // Step 3: Click the login button
    await loginButton.click();

    // Step 4: Wait for navigation and verify the redirection URL
    await page.waitForURL('http://127.0.0.1:5000/');

    await page.waitForTimeout(3000);
});

test('Start Detection button functionality and video feed capture', async ({ page }) => {
    // Navigate to the main page and login
    await page.goto('http://127.0.0.1:5000/login');
    const usernameInput = page.locator(locators.loginPage.usernameInput);
    const loginButton = page.locator(locators.loginPage.loginButton);
    const startBtn = page.locator(locators.mainPage.startBtn);
    const stopBtn = page.locator(locators.mainPage.stopBtn);
    const videoElement = page.locator(locators.mainPage.videoElement);
    await usernameInput.fill('Sonal');
    await loginButton.click();

    // Wait for the main page to load
    await page.waitForURL('http://127.0.0.1:5000/');
  
    // Verify start button is initially enabled and stop button is disabled
    await expect(startBtn).toBeEnabled();
    await expect(stopBtn).toBeDisabled();

    // Click start button
    await startBtn.click();

    // Wait for video feed to start
    await page.waitForTimeout(1000);

    // Verify video source is not empty
    const videoSrc = await videoElement.getAttribute('src');
    expect(videoSrc).toBeTruthy();

    // Verify buttons state after starting video
    await expect(startBtn).toBeDisabled();
    await expect(stopBtn).toBeEnabled();

    await page.waitForTimeout(3000);

    await expect(videoElement).toBeVisible();
});

test('Stop Detection button functionality', async ({ page }) => {
    // Navigate to the login page and login
    await page.goto('http://127.0.0.1:5000/login');
    const usernameInput = page.locator(locators.loginPage.usernameInput);
    const loginButton = page.locator(locators.loginPage.loginButton);
    await usernameInput.fill('Sonal');
    await loginButton.click();

    // Wait for main page to load
    await page.waitForURL('http://127.0.0.1:5000/');

    // Locate video elements and buttons
    const startBtn = page.locator(locators.mainPage.startBtn);
    const stopBtn = page.locator(locators.mainPage.stopBtn);
    const videoElement = page.locator(locators.mainPage.videoElement);
    const analyzeBtn = page.locator(locators.mainPage.analyzeBtn);

    // Start video detection
    await startBtn.click();

    // Verify video is streaming
    const videoSrcBefore = await videoElement.getAttribute('src');
    expect(videoSrcBefore).toBeTruthy();
    await page.waitForTimeout(7000);

    // Click stop button
    await stopBtn.click();
    await page.waitForTimeout(2000);

    // Verify button states after stopping
    await expect(startBtn).toBeEnabled();
    await expect(stopBtn).toBeDisabled();
    await expect(analyzeBtn).toBeEnabled();

    // Verify video source is cleared
    const videoSrcAfter = await videoElement.getAttribute('src');
    expect(videoSrcAfter).toBe('');
});

test('Analyze button functionality', async ({ page }) => {
    // Navigate to the login page and login
    await page.goto('http://127.0.0.1:5000/login');
    const usernameInput = page.locator(locators.loginPage.usernameInput);
    const loginButton = page.locator(locators.loginPage.loginButton);
    await usernameInput.fill('Sonal');
    await loginButton.click();

    // Wait for main page to load
    await page.waitForURL('http://127.0.0.1:5000/');

    // Locate elements
    const startBtn = page.locator(locators.mainPage.startBtn);
    const stopBtn = page.locator(locators.mainPage.stopBtn);
    const analyzeBtn = page.locator(locators.mainPage.analyzeBtn);
    const analysisResults = page.locator(locators.analysisResults);
    const downloadReportBtn = page.locator(locators.mainPage.downloadReportBtn);

    // Start and stop video detection
    await startBtn.click();
    await page.waitForTimeout(7000);
    await stopBtn.click();

    // Click Analyze button
    await analyzeBtn.click();
    await page.waitForTimeout(10000);

    // Verify analysis results are populated
    const resultsText = await analysisResults.textContent();
    expect(resultsText).not.toBe('No analysis results yet');

    // Verify download report button is enabled
    await expect(downloadReportBtn).toBeEnabled();
});

test('Logout form validation and redirection', async ({ page }) => {
    // Navigate to the login page and login
    await page.goto('http://127.0.0.1:5000/login');
    const usernameInput = page.locator(locators.loginPage.usernameInput);
    const loginButton = page.locator(locators.loginPage.loginButton);
    await usernameInput.fill('Sonal');
    await loginButton.click();

    // Wait for main page to load
    await page.waitForURL('http://127.0.0.1:5000/');

    // Locate and verify logout button
    const logoutButton = page.locator(locators.mainPage.logoutButton);
    await expect(logoutButton).toBeVisible();

    await page.waitForTimeout(1000);

    // Click logout button
    await logoutButton.click();

    // Verify redirection to login page
    await page.waitForURL('http://127.0.0.1:5000/login');

    await page.waitForTimeout(3000);
});
