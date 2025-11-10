# Gemini API Setup Guide

## Problem: 404 Model Not Found

The Gemini API was returning 404 errors because `gemini-1.5-pro` and `gemini-1.5-flash` were **deprecated in September 2025**. These models are no longer available.

## Solution: Use Newer Models

### Step 1: Get Your API Key

1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with your Google account
3. Click "Create API Key" or use an existing one
4. Copy the API key

### Step 2: Update Your .env File

Add or update these lines in `backend/.env`:

```env
GEMINI_API_KEY=your_api_key_here
GEMINI_MODEL=gemini-2.0-flash-exp
```

### Step 3: Available Models (as of 2025)

The following models are currently available:

- `gemini-2.0-flash-exp` (Recommended - Latest experimental)
- `gemini-2.0-flash-thinking-exp` (Latest with thinking capabilities)
- `gemini-1.5-pro-latest` (Latest stable Pro model)
- `gemini-1.5-flash-latest` (Latest stable Flash model)
- `gemini-pro` (Fallback option)

### Step 4: Verify Your Setup

1. Restart your backend server
2. Check the logs for:
   - "Gemini model 'gemini-2.0-flash-exp' initialized successfully"
   - Or: "Available Gemini models: ..."

If you see errors:
- **401/403**: Your API key is invalid. Check that it's correctly set in `.env`
- **404**: The model name is incorrect. Try one of the models listed above
- **No API key error**: Make sure `GEMINI_API_KEY` is set in your `.env` file

### Step 5: Test the API

The application will automatically:
1. Try to use the configured model
2. If it fails, list available models
3. Try alternative models automatically
4. Fall back to template-based notes if all models fail

## Troubleshooting

### Check Available Models Programmatically

The code now includes a function to list available models. Check the logs when the server starts to see which models are available for your API key.

### API Key Permissions

Make sure your API key has the necessary permissions:
- Go to [Google Cloud Console](https://console.cloud.google.com/)
- Navigate to "APIs & Services" > "Credentials"
- Check that your API key has "Generative Language API" enabled

### Rate Limits

Free tier API keys have rate limits. If you hit rate limits:
- Wait a few minutes and try again
- Consider upgrading to a paid plan for higher limits

## Fallback Behavior

If Gemini is unavailable, the application will:
- Use template-based note generation
- Extract information from transcription using pattern matching
- Still generate useful SOAP notes, though they may be less detailed than AI-generated ones

This ensures the application continues to work even if the Gemini API is temporarily unavailable.

