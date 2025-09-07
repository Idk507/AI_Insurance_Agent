# Azure AD SSO Setup Guide for Agent BuddyBee

This guide will help you set up Single Sign-On (SSO) authentication using Microsoft Azure Active Directory for the Agent BuddyBee application.

## Prerequisites

- Python 3.8+ installed
- Azure Active Directory tenant
- Administrator access to register applications in Azure AD

## Configuration Steps

### 1. Create Environment File

Create a `.env` file in the project root directory with the following content:

```env
# MSAL Azure AD Configuration
MSAL_CLIENT_ID=f8df195a-151f-44df-88ce-215640d63afe
MSAL_TENANT_ID=9f2e2b7d-6081-4b3a-b7f5-433b581f6a5f
MSAL_CLIENT_SECRET=MGv8Q~.B3LGdDt9bmqYAcoLtRXHQ3hyzudOm7a_V
MSAL_AUTHORITY=https://login.microsoftonline.com/9f2e2b7d-6081-4b3a-b7f5-433b581f6a5f

# Session Configuration
SESSION_SECRET=your-very-secure-random-session-secret-key-here

# Application Configuration
REDIRECT_URI=http://localhost:8000/auth/callback
```

**Important**: Replace `SESSION_SECRET` with a secure random string. You can generate one using:
```python
import secrets
print(secrets.token_urlsafe(32))
```

### 2. Azure AD App Registration

If you need to register your own Azure AD application:

1. Go to [Azure Portal](https://portal.azure.com)
2. Navigate to Azure Active Directory > App registrations
3. Click "New registration"
4. Configure the application:
   - Name: Agent BuddyBee
   - Supported account types: Choose based on your requirements
   - Redirect URI: `http://localhost:8000/auth/callback` (for development)
5. After registration, note down:
   - Application (client) ID
   - Directory (tenant) ID
6. Go to "Certificates & secrets" and create a new client secret
7. Update the `.env` file with your values

### 3. Configure Redirect URIs

In your Azure AD app registration:
1. Go to "Authentication"
2. Add the following redirect URIs:
   - `http://localhost:8000/auth/callback` (development)
   - `https://beeonixagent.bigtapp.net/auth/callback` (production)
3. Enable "ID tokens" and "Access tokens" under "Implicit grant and hybrid flows"

### 4. Install Dependencies

Make sure all required packages are installed:

```bash
pip install -r requirements.txt
```

### 5. Run the Application

Start the application:

```bash
python -m uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

### 6. Access the Application

1. Open your browser and navigate to `http://localhost:8000`
2. You will see the login page
3. Click "Sign in with Microsoft"
4. Enter your Azure AD credentials
5. After successful authentication, you'll be redirected to the main application

## Troubleshooting

### Common Issues

1. **"Invalid client secret" error**
   - Verify the client secret in your `.env` file
   - Check if the secret has expired in Azure AD

2. **"Redirect URI mismatch" error**
   - Ensure the redirect URI in Azure AD matches exactly with your application
   - Check for http vs https differences

3. **Session errors**
   - Make sure `SESSION_SECRET` is set in your `.env` file
   - Restart the application after changing environment variables

4. **CORS errors**
   - Add your domain to the CORS configuration in `app.py`

### Debug Mode

To enable debug logging, set the logging level in `app.py`:

```python
logging.basicConfig(level=logging.DEBUG)
```

## Security Considerations

1. **Never commit credentials**: Keep your `.env` file in `.gitignore`
2. **Use HTTPS in production**: Update redirect URIs accordingly
3. **Rotate secrets regularly**: Update client secrets periodically
4. **Implement proper session management**: Set appropriate session timeouts

## Production Deployment

For production deployment:

1. Update the redirect URI to use HTTPS
2. Set secure session configuration
3. Use environment variables from your hosting platform
4. Enable additional security headers
5. Configure proper logging and monitoring

## API Permissions

The application requests the following Microsoft Graph API permissions:
- `User.Read`: Read user profile
- `User.ReadBasic.All`: Read basic profiles of all users

## Support

For issues or questions:
1. Check the application logs in `insurance_assistant.log`
2. Verify Azure AD configuration
3. Ensure all environment variables are properly set 