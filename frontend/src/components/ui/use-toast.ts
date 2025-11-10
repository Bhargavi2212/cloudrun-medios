type ToastVariant = 'default' | 'destructive'

type ToastArgs = {
  title: string
  description?: string
  variant?: ToastVariant
}

export const useToast = () => {
  const toast = ({ title, description }: ToastArgs) => {
    const message = description ? `${title}

${description}` : title
    if (typeof window !== 'undefined') {
      window.alert(message)
    } else {
      // eslint-disable-next-line no-console
      console.log(message)
    }
  }

  return { toast }
}
