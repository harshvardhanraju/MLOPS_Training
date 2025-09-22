#!/usr/bin/env python3
"""
Convert MLOps Workshop Markdown presentation to professional PDF
"""

import os
import sys
from pathlib import Path
import markdown
from weasyprint import HTML, CSS
from weasyprint.text.fonts import FontConfiguration

def create_pdf_from_markdown():
    """Convert markdown presentation to professional PDF"""

    # File paths
    md_file = "docs/presentations/MLOps_Workshop_Complete.md"
    html_file = "docs/presentations/MLOps_Workshop_Complete.html"
    pdf_file = "docs/presentations/MLOps_Workshop_Complete.pdf"

    print("🔄 Converting MLOps Workshop presentation to PDF...")

    # Read markdown content
    with open(md_file, 'r', encoding='utf-8') as f:
        md_content = f.read()

    # Convert markdown to HTML
    md = markdown.Markdown(extensions=[
        'toc',
        'codehilite',
        'fenced_code',
        'tables',
        'attr_list'
    ])

    html_content = md.convert(md_content)

    # Create professional HTML template
    html_template = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>MLOps Workshop: Complete Pipeline Implementation</title>
        <style>
            @page {{
                size: A4;
                margin: 2cm 1.5cm;
                @top-center {{
                    content: "MLOps Workshop - Complete Pipeline Implementation";
                    font-size: 10pt;
                    color: #666;
                    border-bottom: 1px solid #ddd;
                    padding-bottom: 5pt;
                }}
                @bottom-center {{
                    content: "Page " counter(page) " of " counter(pages);
                    font-size: 10pt;
                    color: #666;
                }}
            }}

            body {{
                font-family: 'Arial', 'Helvetica', sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 100%;
                margin: 0;
                padding: 0;
            }}

            h1 {{
                color: #2c3e50;
                border-bottom: 3px solid #3498db;
                padding-bottom: 10px;
                page-break-before: always;
                font-size: 28pt;
                margin-top: 0;
            }}

            h1:first-of-type {{
                page-break-before: avoid;
                text-align: center;
                border-bottom: none;
                color: #e74c3c;
                font-size: 36pt;
                margin-bottom: 20pt;
            }}

            h2 {{
                color: #34495e;
                border-left: 4px solid #3498db;
                padding-left: 15px;
                margin-top: 30px;
                font-size: 20pt;
                page-break-after: avoid;
            }}

            h3 {{
                color: #2980b9;
                margin-top: 25px;
                font-size: 16pt;
                page-break-after: avoid;
            }}

            h4 {{
                color: #27ae60;
                margin-top: 20px;
                font-size: 14pt;
                page-break-after: avoid;
            }}

            p {{
                margin-bottom: 12px;
                text-align: justify;
                font-size: 11pt;
            }}

            ul, ol {{
                margin-bottom: 15px;
                padding-left: 25px;
                font-size: 11pt;
            }}

            li {{
                margin-bottom: 5px;
            }}

            code {{
                background-color: #f8f9fa;
                padding: 2px 6px;
                border-radius: 3px;
                font-family: 'Courier New', monospace;
                font-size: 10pt;
                border: 1px solid #e9ecef;
            }}

            pre {{
                background-color: #f8f9fa;
                padding: 15px;
                border-radius: 5px;
                border: 1px solid #e9ecef;
                overflow-x: auto;
                font-size: 9pt;
                line-height: 1.4;
                margin: 15px 0;
                page-break-inside: avoid;
            }}

            pre code {{
                background-color: transparent;
                padding: 0;
                border: none;
                font-size: 9pt;
            }}

            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 15px 0;
                font-size: 10pt;
                page-break-inside: avoid;
            }}

            th, td {{
                border: 1px solid #ddd;
                padding: 8px;
                text-align: left;
            }}

            th {{
                background-color: #3498db;
                color: white;
                font-weight: bold;
            }}

            tr:nth-child(even) {{
                background-color: #f9f9f9;
            }}

            blockquote {{
                border-left: 4px solid #3498db;
                margin: 20px 0;
                padding: 10px 20px;
                background-color: #f8f9fa;
                font-style: italic;
            }}

            .advantages {{
                background-color: #d4edda;
                border: 1px solid #c3e6cb;
                border-radius: 5px;
                padding: 10px;
                margin: 10px 0;
            }}

            .disadvantages {{
                background-color: #f8d7da;
                border: 1px solid #f5c6cb;
                border-radius: 5px;
                padding: 10px;
                margin: 10px 0;
            }}

            .architecture-diagram {{
                background-color: #f0f0f0;
                border: 2px solid #3498db;
                border-radius: 10px;
                padding: 20px;
                margin: 20px 0;
                font-family: 'Courier New', monospace;
                font-size: 9pt;
                line-height: 1.2;
                text-align: center;
                page-break-inside: avoid;
            }}

            .toc {{
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 5px;
                padding: 20px;
                margin: 20px 0;
                page-break-after: always;
            }}

            .toc h2 {{
                margin-top: 0;
                border: none;
                padding: 0;
                color: #2c3e50;
            }}

            .toc ul {{
                list-style-type: none;
                padding-left: 0;
            }}

            .toc li {{
                margin-bottom: 8px;
            }}

            .toc a {{
                text-decoration: none;
                color: #3498db;
                font-weight: bold;
            }}

            .section-break {{
                page-break-before: always;
            }}

            .no-break {{
                page-break-inside: avoid;
            }}

            .highlight {{
                background-color: #fff3cd;
                border: 1px solid #ffeaa7;
                border-radius: 5px;
                padding: 15px;
                margin: 15px 0;
            }}

            .warning {{
                background-color: #f8d7da;
                border: 1px solid #f5c6cb;
                border-radius: 5px;
                padding: 15px;
                margin: 15px 0;
            }}

            .info {{
                background-color: #d1ecf1;
                border: 1px solid #bee5eb;
                border-radius: 5px;
                padding: 15px;
                margin: 15px 0;
            }}

            .footer {{
                position: fixed;
                bottom: 0;
                width: 100%;
                text-align: center;
                font-size: 10pt;
                color: #666;
            }}

            /* Specific styling for code blocks */
            .language-python, .language-yaml, .language-bash, .language-dockerfile {{
                background-color: #2c3e50;
                color: #ecf0f1;
            }}

            .language-python code, .language-yaml code, .language-bash code, .language-dockerfile code {{
                color: #ecf0f1;
            }}

            /* Style architecture diagrams */
            pre:contains("┌"), pre:contains("│"), pre:contains("└") {{
                background-color: #f0f8ff;
                border: 2px solid #3498db;
                font-family: 'Courier New', monospace;
                font-size: 8pt;
                text-align: left;
                page-break-inside: avoid;
            }}
        </style>
    </head>
    <body>
        {html_content}
    </body>
    </html>
    """

    # Save HTML file
    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(html_template)

    print(f"✅ HTML file created: {html_file}")

    try:
        # Convert HTML to PDF
        font_config = FontConfiguration()

        css = CSS(string="""
            @page {
                size: A4;
                margin: 2cm 1.5cm;
            }
        """, font_config=font_config)

        html_doc = HTML(filename=html_file)
        html_doc.write_pdf(pdf_file, stylesheets=[css], font_config=font_config)

        print(f"✅ PDF created successfully: {pdf_file}")

        # Get file size
        pdf_size = os.path.getsize(pdf_file) / (1024 * 1024)  # MB
        print(f"📄 PDF size: {pdf_size:.2f} MB")

        return True

    except Exception as e:
        print(f"❌ Error creating PDF: {e}")
        print("💡 Note: PDF creation requires weasyprint. Install with: pip install weasyprint")
        print("📝 HTML version is available for manual conversion or viewing in browser")
        return False

def create_presentation_summary():
    """Create a brief summary document"""
    summary_content = """# MLOps Workshop Presentation Summary

## 📊 Presentation Overview

**Title**: MLOps Workshop: Complete Pipeline Implementation
**Duration**: 3-4 hours (with breaks)
**Audience**: Data Scientists, ML Engineers, DevOps Engineers
**Level**: Intermediate to Advanced

## 🎯 Learning Objectives

By the end of this workshop, participants will:
- Understand complete MLOps pipeline architecture
- Know how to implement 8 essential MLOps tools
- Learn production deployment strategies
- Understand monitoring and observability practices

## 📚 Components Covered

### 1. Data Versioning (DVC)
- **What**: Git-like versioning for data and ML models
- **Why**: Reproducibility and data pipeline management
- **How**: Hands-on DVC setup and pipeline creation

### 2. Model Versioning & Experiment Tracking (MLflow)
- **What**: Complete ML lifecycle management platform
- **Why**: Experiment tracking, model registry, deployment
- **How**: MLflow tracking setup and model registration

### 3. Model Serving (FastAPI)
- **What**: Modern Python web framework for APIs
- **Why**: High-performance model serving with automatic docs
- **How**: REST API creation for model inference

### 4. Monitoring & Observability (Prometheus + Grafana)
- **What**: Metrics collection and visualization stack
- **Why**: Production monitoring and alerting
- **How**: Metrics setup and dashboard creation

### 5. CI/CD Pipelines (GitHub Actions)
- **What**: Automated testing and deployment pipelines
- **Why**: Continuous integration and deployment
- **How**: GitHub Actions workflow setup

### 6. Containerization (Docker)
- **What**: Application containerization platform
- **Why**: Consistent environments and easy deployment
- **How**: Dockerfile creation and Docker Compose orchestration

### 7. Data Drift Detection (Evidently AI)
- **What**: ML model and data quality monitoring
- **Why**: Detect model degradation and data changes
- **How**: Drift detection setup and reporting

### 8. Automated Testing
- **What**: Comprehensive testing for ML systems
- **Why**: Quality assurance and regression prevention
- **How**: Unit, integration, and performance testing

## 🏗️ Architecture Highlights

### Overall System Architecture
- 4-layer architecture: Data, Training, Serving, Monitoring
- Microservices-based design with Docker containers
- Event-driven architecture for automation
- Comprehensive monitoring and observability

### Key Integration Patterns
- Event-driven model retraining
- Circuit breaker for model fallback
- Blue-green deployment strategy
- Multi-layer monitoring approach

## 💡 Best Practices Covered

### Technical Best Practices
- Version everything (data, code, models, configs)
- Implement comprehensive testing strategies
- Use gradual rollouts for deployments
- Monitor at multiple layers (infra, app, business)

### Organizational Best Practices
- Foster collaboration between teams
- Define clear roles and responsibilities
- Invest in team training and documentation
- Start small and evolve incrementally

## 🚀 Implementation Roadmap

### Phase 1: Foundation (Weeks 1-4)
- Version control setup
- Basic CI/CD pipeline
- Model training pipeline
- MLflow experiment tracking

### Phase 2: Serving & Monitoring (Weeks 5-8)
- Model serving with FastAPI
- Prometheus monitoring
- Grafana dashboards
- Automated testing

### Phase 3: Advanced Features (Weeks 9-12)
- Data drift detection
- Model retraining automation
- Security enhancements
- Performance optimization

### Phase 4: Production Hardening (Weeks 13-16)
- Load testing and optimization
- Disaster recovery planning
- Documentation and training
- Compliance and governance

## 📋 Workshop Materials

### Code Repository
- Complete working MLOps demo
- All configuration files
- Comprehensive test suites
- Documentation and guides

### Presentation Formats
- **PDF**: Professional presentation format
- **HTML**: Interactive web version
- **Markdown**: Source format for customization

## 🎓 Next Steps for Participants

### Immediate Actions
1. Clone the demo repository
2. Set up local development environment
3. Run through the complete pipeline
4. Experiment with different models

### Advanced Learning
- Explore additional MLOps platforms (Kubeflow, Seldon)
- Learn about feature stores and model mesh
- Study edge deployment strategies
- Join MLOps communities and forums

## 📞 Support and Resources

### Documentation
- Complete setup guides in docs/guides/
- Architecture documentation in docs/architecture/
- API documentation generated automatically

### Community Resources
- MLOps Community Slack
- GitHub Discussions on the demo repository
- Regular office hours for questions

---

**Ready to transform your ML workflows with production-grade MLOps?**
Let's build reliable, scalable ML systems together! 🚀
"""

    with open("docs/presentations/MLOps_Workshop_Summary.md", 'w') as f:
        f.write(summary_content)

    print("📋 Workshop summary created: docs/presentations/MLOps_Workshop_Summary.md")

def main():
    """Main function to create presentation materials"""
    print("🎯 Creating MLOps Workshop Presentation Materials")
    print("=" * 60)

    # Ensure directories exist
    os.makedirs("docs/presentations", exist_ok=True)

    # Create presentation summary
    create_presentation_summary()

    # Try to create PDF
    pdf_created = create_pdf_from_markdown()

    print("\n" + "=" * 60)
    print("✅ MLOps Workshop Presentation Materials Created!")
    print("\n📁 Available Files:")
    print("   • docs/presentations/MLOps_Workshop_Complete.md (Source)")
    print("   • docs/presentations/MLOps_Workshop_Complete.html (Web version)")
    if pdf_created:
        print("   • docs/presentations/MLOps_Workshop_Complete.pdf (Professional PDF)")
    print("   • docs/presentations/MLOps_Workshop_Summary.md (Quick reference)")

    print("\n🎯 Presentation Highlights:")
    print("   • Complete MLOps pipeline coverage")
    print("   • 8 essential tools with hands-on examples")
    print("   • Production-ready architecture patterns")
    print("   • Best practices and implementation roadmap")
    print("   • Professional formatting for workshops")

    if not pdf_created:
        print("\n💡 To create PDF version:")
        print("   pip install weasyprint")
        print("   python scripts/convert_presentation_to_pdf.py")

if __name__ == "__main__":
    main()