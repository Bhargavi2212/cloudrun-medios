# MediOS Frontend - AI Medical Scribe

A modern React-based frontend for the MediOS AI Medical Scribe system. This application provides an intuitive interface for doctors to record medical notes using voice and get AI-generated clinical documentation.

## Features

- 🎤 **Voice Recording**: Record medical notes using your microphone
- 🤖 **AI-Powered Transcription**: Real-time speech-to-text conversion
- 🧠 **Intelligent Note Generation**: AI-generated clinical notes using TinyLlama
- 📝 **Note Editing**: Edit and refine generated notes before saving
- 💾 **Database Integration**: Save notes directly to patient records
- 🎯 **Patient Management**: Select patients and appointments
- 📊 **Confidence Scoring**: Real-time confidence metrics for AI generation
- 🔄 **Real-time Processing**: Live status updates during processing

## Tech Stack

- **React 18** with TypeScript
- **Vite** for fast development and building
- **Tailwind CSS** for styling
- **Radix UI** for accessible components
- **Lucide React** for icons
- **React Hooks** for state management

## Getting Started

### Prerequisites

- Node.js 18+ 
- npm or yarn
- Backend server running (see backend README)

### Installation

1. Navigate to the frontend directory:
```bash
cd MediOS/frontend
```

2. Install dependencies:
```bash
npm install
```

3. Start the development server:
```bash
npm run dev
```

The application will be available at `http://localhost:3000`

### Building for Production

```bash
npm run build
```

## Project Structure

```
src/
├── components/
│   ├── AIScribe.tsx          # Main AI Scribe component
│   ├── ui/                   # Reusable UI components
│   │   ├── button.tsx
│   │   ├── card.tsx
│   │   ├── input.tsx
│   │   ├── label.tsx
│   │   ├── select.tsx
│   │   ├── textarea.tsx
│   │   ├── toast.tsx
│   │   └── toaster.tsx
├── hooks/
│   └── use-toast.ts          # Toast notification hook
├── lib/
│   └── utils.ts              # Utility functions
├── pages/
│   └── AIScribePage.tsx      # Main page component
├── App.tsx                   # Root application component
├── main.tsx                  # Application entry point
└── index.css                 # Global styles
```

## API Integration

The frontend communicates with the backend through the following endpoints:

- `POST /api/v1/ai/process-audio` - Process audio and generate notes
- `POST /api/v1/notes` - Save notes to database
- `GET /api/v1/patients` - Get patient list
- `GET /api/v1/appointments/patient/{id}` - Get patient appointments

## Usage

1. **Select Patient**: Choose a patient from the dropdown
2. **Start Recording**: Click the microphone button to begin recording
3. **Speak Clearly**: Describe the patient's symptoms, examination findings, and assessment
4. **Stop Recording**: Click the stop button when finished
5. **Review Note**: The AI will generate a clinical note
6. **Edit if Needed**: Make any necessary corrections
7. **Save Note**: Click "Save Note" to store in the patient record

## Development

### Adding New Components

1. Create component in `src/components/`
2. Add TypeScript interfaces for props
3. Use Tailwind CSS for styling
4. Export from component file

### Styling Guidelines

- Use Tailwind CSS utility classes
- Follow the design system in `tailwind.config.js`
- Use CSS variables for theming
- Ensure responsive design

### State Management

- Use React hooks for local state
- Keep state as close to where it's used as possible
- Use context for global state if needed

## Troubleshooting

### Common Issues

1. **Microphone Permission**: Ensure browser has microphone access
2. **API Connection**: Verify backend server is running on port 8000
3. **CORS Issues**: Check backend CORS configuration
4. **Build Errors**: Clear node_modules and reinstall dependencies

### Debug Mode

Enable debug logging by setting `localStorage.debug = 'medios:*'` in browser console.

## Contributing

1. Follow TypeScript best practices
2. Add proper error handling
3. Write meaningful commit messages
4. Test on different browsers
5. Ensure accessibility compliance

## License

This project is part of the MediOS healthcare system.
