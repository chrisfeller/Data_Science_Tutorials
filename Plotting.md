### Plotting Cheat Sheet

---
#### Seaborn Color Palettes
```
moneyball = ["#6cdae7", "#fd3a4a", "#ffaa1d", "#ff23e5", "#34495e", "#2ecc71", "#3498db"]
paired = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71", "#ffaa1d"]
muted = ['#4878d0', '#ee854a', '#6acc64', '#d65f5f', '#956cb4', '#8c613c', '#dc7ec0']

# Global Setting
sns.set_palette(moneyball)

# In an Individual Plot
sns.lineplot(x='Year', y='Count', hue='Position', data=df, palette=moneyball)
```
