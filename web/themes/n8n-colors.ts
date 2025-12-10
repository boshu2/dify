/**
 * n8n Color Palette for Tailwind CSS
 *
 * These colors are based on n8n's brand guidelines and can be used
 * alongside the existing Dify color system.
 */

export const n8nColors = {
  // Primary Orange Scale
  'n8n-orange': {
    50: '#fef7f5',
    100: '#fee3dd',
    200: '#fcc5b9',
    300: '#f99e88',
    400: '#F96E49',
    500: '#EE4F27',
    600: '#B83617',
    700: '#8a280f',
    800: '#5c1a0a',
    900: '#2e0d05',
    DEFAULT: '#EE4F27',
  },

  // Purple Accent Scale
  'n8n-purple': {
    50: '#f6f2ff',
    100: '#ebe4ff',
    200: '#d4c7ff',
    300: '#b49bff',
    400: '#8f5fff',
    500: '#6B21EF',
    600: '#5b1bcf',
    700: '#4a15a8',
    800: '#3a1082',
    900: '#2a0b5c',
    DEFAULT: '#6B21EF',
  },

  // Navy Background Scale (for dark mode)
  'n8n-navy': {
    50: '#f5f4f7',
    100: '#e8e6ec',
    200: '#C4BBD3',
    300: '#7c7291',
    400: '#4a4259',
    500: '#342c3e',
    600: '#252233',
    700: '#1F192A',
    800: '#1B1728',
    900: '#0E0918',
    DEFAULT: '#1F192A',
  },

  // n8n Green (Success)
  'n8n-green': {
    50: '#f0fdf6',
    100: '#dcfce9',
    200: '#bbf7d4',
    300: '#86efb3',
    400: '#47cd89',
    500: '#35A670',
    600: '#186F44',
    700: '#145c39',
    800: '#12492e',
    900: '#0f3926',
    DEFAULT: '#35A670',
  },

  // Workflow Canvas Colors
  'n8n-canvas': {
    bg: '#0E0918',
    grid: '#7e7e7e',
    'node-bg': '#342c3e',
    'node-border': 'rgba(255, 255, 255, 0.1)',
    'node-selected': '#6B21EF',
    edge: '#6B21EF',
    'edge-animated': '#EE4F27',
  },
}

// Tailwind CSS variable references for the n8n theme
export const n8nThemeVars = {
  // Workflow-specific variables
  'workflow-canvas-bg': 'var(--color-workflow-canvas-bg)',
  'workflow-canvas-grid': 'var(--color-workflow-canvas-grid)',
  'workflow-node-bg': 'var(--color-workflow-node-bg)',
  'workflow-node-bg-hover': 'var(--color-workflow-node-bg-hover)',
  'workflow-node-border': 'var(--color-workflow-node-border)',
  'workflow-node-border-selected': 'var(--color-workflow-node-border-selected)',
  'workflow-node-shadow': 'var(--color-workflow-node-shadow)',
  'workflow-edge': 'var(--color-workflow-edge)',
  'workflow-edge-animated': 'var(--color-workflow-edge-animated)',
  'workflow-minimap-bg': 'var(--color-workflow-minimap-bg)',
  'workflow-minimap-node': 'var(--color-workflow-minimap-node)',
}

export default n8nColors
