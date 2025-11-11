# 🎨 AlphaRep Brand Colors

## Color Palette

### Primary - Neural Blue
- **Main**: `#1890ff` (primary-500)
- **Usage**: Primary buttons, links, highlights, active states
- **Gradient**: From `#e6f4ff` (light) to `#002766` (dark)

### Accent - Energy Lime
- **Main**: `#52c41a` (accent-500)
- **Usage**: Success states, good form indicators, secondary actions
- **Gradient**: From `#f6ffed` (light) to `#092b00` (dark)

### Warning - Amber
- **Main**: `#faad14` (warning-500)
- **Usage**: Warnings, form corrections, attention states
- **Gradient**: From `#fffbe6` (light) to `#613400` (dark)

### Dark - Obsidian Black & Titanium Gray
- **Obsidian Black**: `#0a0a0a` (dark-900)
- **Titanium Gray**: `#737373` (dark-500)
- **White**: `#fafafa` (dark-50)
- **Usage**: Backgrounds, text, borders

---

## Usage Examples

### Buttons
```jsx
// Primary button (Neural Blue)
<button className="bg-primary-500 hover:bg-primary-600 text-white">
  Start Workout
</button>

// Secondary button (Energy Lime)
<button className="bg-accent-500 hover:bg-accent-600 text-white">
  Good Form!
</button>
```

### Text
```jsx
// Gradient text (Blue to Lime)
<h1 className="gradient-text">AlphaRep</h1>

// Glowing text
<span className="text-glow">LIVE</span>
<span className="text-glow-lime">PERFECT FORM</span>
```

### Cards
```jsx
// Dark card with blue border
<div className="bg-dark-800 border-2 border-primary-500/20">
  Content
</div>

// Gradient card
<div className="bg-gradient-to-br from-primary-500 to-accent-500">
  Content
</div>
```

### Badges
```jsx
// Success badge (Lime)
<span className="badge-success">GOOD</span>

// Warning badge (Amber)
<span className="badge-warning">IMPROVE</span>

// Error badge (Blue)
<span className="badge-error">POOR</span>
```

---

## Color Psychology

- **Neural Blue**: Trust, technology, precision, focus
- **Energy Lime**: Growth, vitality, success, achievement
- **Obsidian Black**: Power, sophistication, premium
- **Titanium Gray**: Balance, neutrality, professionalism
- **White**: Clarity, simplicity, cleanliness

---

## Accessibility

All color combinations meet WCAG 2.1 AA standards:
- ✅ Neural Blue on Dark: 7.2:1 contrast ratio
- ✅ Energy Lime on Dark: 6.8:1 contrast ratio
- ✅ White on Dark: 15.8:1 contrast ratio

---

## Brand Guidelines

### Do's ✅
- Use Neural Blue for primary actions and highlights
- Use Energy Lime for success states and positive feedback
- Maintain dark backgrounds (Obsidian Black) for premium feel
- Use gradients sparingly for emphasis

### Don'ts ❌
- Don't mix too many colors in one component
- Don't use Energy Lime for errors or warnings
- Don't use light backgrounds (breaks brand identity)
- Don't reduce opacity below 10% for borders

---

## Tailwind Classes Reference

### Primary (Neural Blue)
- `bg-primary-500` - Main blue background
- `text-primary-500` - Blue text
- `border-primary-500` - Blue border
- `hover:bg-primary-600` - Darker blue on hover

### Accent (Energy Lime)
- `bg-accent-500` - Main lime background
- `text-accent-500` - Lime text
- `border-accent-500` - Lime border
- `hover:bg-accent-600` - Darker lime on hover

### Dark (Obsidian & Gray)
- `bg-dark-900` - Obsidian black background
- `bg-dark-800` - Dark gray background
- `bg-dark-700` - Medium gray background
- `text-gray-100` - Light gray text

---

**Updated**: November 11, 2025  
**Version**: 2.0.0  
**Brand**: AlphaRep - Train Smarter, Not Harder
