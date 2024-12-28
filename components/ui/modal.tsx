import * as React from 'react';
import { Dialog, DialogContent, DialogOverlay } from '@reach/dialog';
import '@reach/dialog/styles.css';

interface ModalProps {
  isOpen: boolean;
  onClose: () => void;
  title: string;
  children: React.ReactNode;
}

export function Modal({ isOpen, onClose, title, children }: ModalProps) {
  return (
    <Dialog isOpen={isOpen} onDismiss={onClose}>
      <DialogOverlay />
      <DialogContent>
        <p>{title}</p>
        {children}
        <button onClick={onClose}>Close</button>
      </DialogContent>
    </Dialog>
  );
}
