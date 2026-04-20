from __future__ import annotations

import random


PYTHON_TASKS = [
    ("todo_api", "Build a Flask todo API with CRUD endpoints."),
    ("fastapi_todo", "Build a FastAPI todo API with CRUD endpoints and Pydantic models."),
    ("fastapi_auth", "Build a FastAPI auth service with login and token verification endpoints."),
    ("csv_summary", "Write a script that reads a CSV and prints summary statistics."),
    ("file_backup", "Create a Python utility that copies files into a backup directory."),
    ("json_config", "Load a JSON config file and validate required keys."),
    ("http_client", "Call a REST endpoint and print parsed JSON results."),
    ("pandas_etl", "Write a Python ETL script using pandas to clean and aggregate records."),
    ("cli_tool", "Create a Python CLI tool using argparse for batch file renaming."),
    ("sqlite_crud", "Build a small Python CRUD service backed by SQLite."),
    ("flask_uploads", "Create a Flask app that accepts file uploads and stores metadata."),
    ("fastapi_pagination", "Build a FastAPI endpoint with filtering, sorting, and pagination."),
    ("email_sender", "Write a Python utility that renders and sends notification emails."),
    ("retry_client", "Implement a Python API client with retries, backoff, and timeout handling."),
    ("yaml_loader", "Load a YAML config file, validate keys, and apply environment overrides."),
    ("metrics_job", "Write a scheduled Python job that computes daily metrics and writes a report."),
    ("data_merge", "Merge multiple JSON files into one cleaned dataset with deduplication."),
    ("image_resize", "Create a Python script that resizes images and writes thumbnails."),
    ("webhook_handler", "Build a lightweight webhook receiver that verifies signatures."),
    ("cache_layer", "Implement a Python cache wrapper with TTL support and cache invalidation."),
    ("queue_worker", "Write a Python worker that polls a queue and processes background jobs."),
    ("report_pdf", "Generate a PDF report from tabular input data in Python."),
    ("log_parser", "Parse application logs and aggregate counts by severity and service."),
    ("batch_import", "Build a batch import tool that validates rows and writes failures to a file."),
    ("auth_middleware", "Create Python middleware that checks API keys and request headers."),
    ("inventory_sync", "Write a Python sync script that reconciles inventory between two systems."),
    ("notification_rules", "Implement Python logic for alert thresholds and escalation rules."),
]

JAVASCRIPT_TASKS = [
    ("login_form", "Write a React login form that posts credentials to an API."),
    ("react_dashboard", "Build a React dashboard component with cards, charts placeholders, and data fetching."),
    ("nextjs_page", "Create a Next.js page with server-side data fetching and a list view."),
    ("nextjs_api", "Create a Next.js API route that returns paginated JSON data."),
    ("todo_ui", "Create a JavaScript todo list with add and delete actions."),
    ("fetch_table", "Fetch a list of users and render them in an HTML table."),
    ("modal_widget", "Build a reusable modal open/close widget."),
    ("express_api", "Write an Express.js REST API with CRUD routes for products."),
    ("node_worker", "Write a Node.js worker script that reads jobs from a queue and processes them."),
    ("kanban_board", "Build a React kanban board with drag-and-drop style task movement."),
    ("settings_page", "Create a React settings page with form validation and save handling."),
    ("search_filters", "Build a JavaScript search UI with debounced filters and sorting."),
    ("upload_widget", "Create a file upload widget with progress display and error handling."),
    ("chart_panel", "Build a dashboard panel that fetches metrics and renders chart-ready data."),
    ("notification_center", "Implement a notification center with unread counts and dismiss actions."),
    ("profile_editor", "Create a profile editor form with optimistic updates."),
    ("checkout_form", "Write a checkout form that validates address and payment fields."),
    ("socket_feed", "Build a small client that consumes websocket events and updates the UI."),
    ("express_auth", "Create an Express auth API with login, refresh token, and logout routes."),
    ("express_uploads", "Write an Express route that handles multipart file uploads."),
    ("cron_reporter", "Build a Node.js script that collects records and sends a summary report."),
    ("admin_table", "Create an admin table with pagination, filtering, and row actions."),
    ("theme_toggle", "Implement a reusable theme toggle with persisted preference."),
    ("wizard_form", "Build a multi-step form wizard with step validation."),
    ("activity_feed", "Render an activity feed with polling and incremental updates."),
    ("cart_store", "Implement a shopping cart store with add, remove, and quantity updates."),
    ("api_proxy", "Create a Node.js proxy endpoint that forwards requests and rewrites headers."),
    ("landing_page", "Build a modern startup landing page with hero, feature grid, CTA, and footer."),
    ("pricing_section", "Create a pricing section component with monthly and annual plan toggles."),
    ("testimonial_strip", "Build a testimonials section with customer quotes, avatars, and logos."),
    ("hero_banner", "Create a SaaS hero banner with headline, subcopy, email capture, and CTA buttons."),
    ("feature_grid", "Build a feature grid section with icons, titles, and short descriptions."),
    ("faq_accordion", "Create an FAQ accordion component for a product landing page."),
    ("navbar_menu", "Build a responsive navbar with dropdown menus and mobile navigation."),
    ("footer_links", "Create a product website footer with columns of links and social icons."),
    ("waitlist_form", "Build a waitlist signup form with optimistic success and validation errors."),
    ("analytics_dashboard", "Create an analytics dashboard with KPI cards, charts placeholders, and filters."),
    ("billing_page", "Build a billing settings page with plan summary, invoices, and payment method form."),
    ("onboarding_flow", "Create a multi-step onboarding flow for a B2B SaaS product."),
    ("team_settings", "Build a team management page with member roles, invites, and access controls."),
    ("marketing_site", "Create a YC-style marketing website for an AI startup with sections for product, trust, and CTA."),
    ("app_sidebar", "Build an application sidebar with nested navigation, badges, and active states."),
    ("command_palette", "Create a command palette modal with keyboard shortcuts and searchable actions."),
    ("empty_state", "Build reusable empty-state components for dashboard pages with CTA actions."),
    ("dashboard_header", "Create a dashboard header with breadcrumbs, date filters, and export buttons."),
    ("notifications_panel", "Build a notifications panel with tabs for all, unread, and archived items."),
    ("integrations_page", "Create an integrations settings page with connected apps and enable toggles."),
    ("docs_search", "Build a documentation search UI with grouped results and highlighted matches."),
    ("ai_chat_ui", "Create an AI chat interface with conversation history, composer, and loading states."),
    ("signup_page", "Build a modern sign-up page with OAuth buttons and inline validation."),
    ("password_reset", "Create a password reset flow with request, verify, and update forms."),
    ("checkout_summary", "Build an ecommerce checkout summary panel with shipping, tax, and total calculations."),
    ("product_grid", "Create a product card grid with filters, badges, and add-to-cart actions."),
    ("admin_metrics", "Build an admin metrics page with tables, status chips, and chart placeholders."),
    ("release_notes", "Create a release notes page with changelog entries and version filtering."),
    ("blog_index", "Build a blog index page with featured posts, categories, and pagination."),
    ("case_studies", "Create a case studies section for a startup website with results cards."),
    ("logo_cloud", "Build a customer logo cloud section with headings and trust copy."),
    ("contact_sales", "Create a contact sales page with lead form, FAQs, and calendar CTA."),
    ("status_page", "Build a status page showing service health, incidents, and maintenance banners."),
    ("roadmap_board", "Create a public roadmap board with tabs for planned, in progress, and shipped items."),
    ("referral_widget", "Build a referral widget with invite link copy, reward summary, and recent referrals."),
]

SHELL_TASKS = [
    ("deploy_script", "Write a bash script that installs dependencies and restarts a service."),
    ("log_cleanup", "Create a shell script that archives and deletes old logs."),
    ("env_setup", "Set environment variables and create project directories."),
    ("backup_tar", "Compress a directory into a timestamped tar.gz archive."),
    ("health_check", "Write a shell script that checks a service health endpoint and exits nonzero on failure."),
    ("rotate_backups", "Create a bash script that keeps the newest backups and deletes older archives."),
    ("sync_files", "Write a shell script that syncs files to a remote host using rsync."),
    ("db_dump", "Create a script that dumps a database and uploads the artifact to object storage."),
    ("release_tag", "Write a shell script that tags a release and prints the latest changelog section."),
    ("csv_scan", "Build a shell pipeline that scans CSV files and reports row counts."),
    ("process_watch", "Write a bash script that watches processes and restarts a worker if it exits."),
    ("disk_alert", "Create a shell script that reports disk usage above a threshold."),
    ("secret_check", "Write a shell script that verifies required environment variables are set."),
    ("artifact_bundle", "Create a script that packages build artifacts into a release directory."),
    ("service_logs", "Write a script that tails service logs and filters for errors."),
    ("cleanup_tmp", "Create a shell script that removes stale files from tmp directories."),
]


def _python_template(task_name: str) -> str:
    if task_name == "todo_api":
        return """from flask import Flask, jsonify, request

app = Flask(__name__)
TODOS = []


@app.get("/todos")
def list_todos():
    return jsonify(TODOS)


@app.post("/todos")
def create_todo():
    payload = request.get_json(force=True)
    item = {"id": len(TODOS) + 1, "title": payload["title"], "done": False}
    TODOS.append(item)
    return jsonify(item), 201


@app.patch("/todos/<int:todo_id>")
def update_todo(todo_id):
    payload = request.get_json(force=True)
    for item in TODOS:
        if item["id"] == todo_id:
            item["done"] = payload.get("done", item["done"])
            return jsonify(item)
    return jsonify({"error": "not found"}), 404


if __name__ == "__main__":
    app.run(debug=True)
"""
    if task_name == "csv_summary":
        return """import csv
from collections import defaultdict


def summarize_csv(path):
    counts = defaultdict(int)
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            for key, value in row.items():
                if value:
                    counts[key] += 1
    for key, value in counts.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    summarize_csv("data.csv")
"""
    if task_name == "file_backup":
        return """from pathlib import Path
import shutil


def backup_files(source_dir, backup_dir):
    source = Path(source_dir)
    target = Path(backup_dir)
    target.mkdir(parents=True, exist_ok=True)
    for path in source.glob("*"):
        if path.is_file():
            shutil.copy2(path, target / path.name)


backup_files("input", "backup")
"""
    if task_name == "json_config":
        return """import json


REQUIRED_KEYS = {"host", "port", "debug"}


def load_config(path):
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    missing = REQUIRED_KEYS - data.keys()
    if missing:
        raise ValueError(f"Missing keys: {sorted(missing)}")
    return data


print(load_config("config.json"))
"""
    if task_name == "fastapi_todo":
        return """from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()


class TodoIn(BaseModel):
    title: str
    done: bool = False


todos: list[dict] = []


@app.get("/todos")
def list_todos():
    return todos


@app.post("/todos")
def create_todo(payload: TodoIn):
    item = {"id": len(todos) + 1, **payload.model_dump()}
    todos.append(item)
    return item


@app.put("/todos/{todo_id}")
def update_todo(todo_id: int, payload: TodoIn):
    for item in todos:
        if item["id"] == todo_id:
            item.update(payload.model_dump())
            return item
    raise HTTPException(status_code=404, detail="Todo not found")
"""
    if task_name == "fastapi_auth":
        return """from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()


class LoginRequest(BaseModel):
    email: str
    password: str


@app.post("/login")
def login(payload: LoginRequest):
    if payload.email == "demo@example.com" and payload.password == "secret":
        return {"access_token": "demo-token", "token_type": "bearer"}
    raise HTTPException(status_code=401, detail="Invalid credentials")


@app.get("/verify")
def verify(token: str):
    return {"valid": token == "demo-token"}
"""
    if task_name == "pandas_etl":
        return """import pandas as pd


def build_daily_summary(input_path: str, output_path: str) -> None:
    frame = pd.read_csv(input_path)
    frame["created_at"] = pd.to_datetime(frame["created_at"])
    summary = (
        frame.assign(day=frame["created_at"].dt.date)
        .groupby("day")["amount"]
        .sum()
        .reset_index()
    )
    summary.to_csv(output_path, index=False)


build_daily_summary("orders.csv", "daily_summary.csv")
"""
    if task_name == "cli_tool":
        return """import argparse
from pathlib import Path


def rename_files(directory: str, prefix: str) -> None:
    for index, path in enumerate(sorted(Path(directory).glob("*"))):
        if path.is_file():
            target = path.with_name(f"{prefix}_{index}{path.suffix}")
            path.rename(target)


parser = argparse.ArgumentParser()
parser.add_argument("directory")
parser.add_argument("--prefix", default="renamed")
args = parser.parse_args()
rename_files(args.directory, args.prefix)
"""
    return """import requests


def fetch_items(url):
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    return response.json()


for item in fetch_items("https://example.com/api/items"):
    print(item)
"""


def _javascript_template(task_name: str) -> str:
    if task_name == "login_form":
        return """import React, { useState } from "react";

export default function LoginForm() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");

  async function handleSubmit(event) {
    event.preventDefault();
    const response = await fetch("/api/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, password })
    });
    const data = await response.json();
    console.log(data);
  }

  return (
    <form onSubmit={handleSubmit}>
      <input value={email} onChange={(e) => setEmail(e.target.value)} />
      <input value={password} onChange={(e) => setPassword(e.target.value)} />
      <button type="submit">Login</button>
    </form>
  );
}
"""
    if task_name == "react_dashboard":
        return """import React, { useEffect, useState } from "react";

export default function Dashboard() {
  const [stats, setStats] = useState([]);

  useEffect(() => {
    async function loadStats() {
      const response = await fetch("/api/stats");
      const data = await response.json();
      setStats(data.items);
    }
    loadStats();
  }, []);

  return (
    <section>
      <h1>Team Dashboard</h1>
      <div className="grid">
        {stats.map((item) => (
          <article key={item.label}>
            <h2>{item.label}</h2>
            <p>{item.value}</p>
          </article>
        ))}
      </div>
    </section>
  );
}
"""
    if task_name == "nextjs_page":
        return """export async function getServerSideProps() {
  const response = await fetch("https://example.com/api/posts");
  const posts = await response.json();
  return { props: { posts } };
}

export default function PostsPage({ posts }) {
  return (
    <main>
      <h1>Posts</h1>
      <ul>
        {posts.map((post) => (
          <li key={post.id}>{post.title}</li>
        ))}
      </ul>
    </main>
  );
}
"""
    if task_name == "nextjs_api":
        return """export default function handler(req, res) {
  const page = Number(req.query.page || 1);
  const items = Array.from({ length: 10 }, (_, index) => ({
    id: (page - 1) * 10 + index + 1,
    name: `Item ${index + 1}`
  }));
  res.status(200).json({ page, items });
}
"""
    if task_name == "todo_ui":
        return """const todos = [];

function addTodo(title) {
  todos.push({ id: Date.now(), title, done: false });
  render();
}

function removeTodo(id) {
  const index = todos.findIndex((item) => item.id === id);
  if (index >= 0) {
    todos.splice(index, 1);
  }
  render();
}

function render() {
  const root = document.getElementById("app");
  root.innerHTML = todos.map((item) => `<li>${item.title}</li>`).join("");
}
"""
    if task_name == "fetch_table":
        return """async function renderUsers() {
  const response = await fetch("/api/users");
  const users = await response.json();
  const rows = users.map((user) => `
    <tr>
      <td>${user.name}</td>
      <td>${user.email}</td>
    </tr>
  `);
  document.querySelector("tbody").innerHTML = rows.join("");
}

renderUsers();
"""
    if task_name == "express_api":
        return """const express = require("express");

const app = express();
app.use(express.json());

const products = [];

app.get("/products", (req, res) => {
  res.json(products);
});

app.post("/products", (req, res) => {
  const item = { id: products.length + 1, ...req.body };
  products.push(item);
  res.status(201).json(item);
});

app.listen(3000, () => {
  console.log("Server running on port 3000");
});
"""
    if task_name == "node_worker":
        return """async function processJob(job) {
  console.log(`Processing ${job.id}`);
  return { id: job.id, status: "done" };
}

async function main() {
  const jobs = [{ id: 1 }, { id: 2 }, { id: 3 }];
  for (const job of jobs) {
    const result = await processJob(job);
    console.log(result);
  }
}

main();
"""
    return """export function createModal(root) {
  const state = { open: false };

  function render() {
    root.innerHTML = state.open ? "<div class='modal'>Hello</div>" : "";
  }

  return {
    open() {
      state.open = true;
      render();
    },
    close() {
      state.open = false;
      render();
    }
  };
}
"""


def _shell_template(task_name: str) -> str:
    if task_name == "deploy_script":
        return """#!/usr/bin/env bash
set -euo pipefail

APP_DIR="/srv/app"

cd "$APP_DIR"
python -m pip install -r requirements.txt
systemctl restart demo-app.service
systemctl status demo-app.service --no-pager
"""
    if task_name == "log_cleanup":
        return """#!/usr/bin/env bash
set -euo pipefail

LOG_DIR="./logs"
ARCHIVE_DIR="./archive"
mkdir -p "$ARCHIVE_DIR"
find "$LOG_DIR" -name "*.log" -mtime +7 -print0 | while IFS= read -r -d '' file; do
  gzip -c "$file" > "$ARCHIVE_DIR/$(basename "$file").gz"
  rm "$file"
done
"""
    if task_name == "env_setup":
        return """#!/usr/bin/env bash
set -euo pipefail

export APP_ENV=development
export APP_PORT=8080
mkdir -p data logs tmp
touch logs/app.log
echo "Environment ready on port $APP_PORT"
"""
    return """#!/usr/bin/env bash
set -euo pipefail

SOURCE_DIR="${1:-data}"
STAMP=$(date +"%Y%m%d_%H%M%S")
tar -czf "backup_${STAMP}.tar.gz" "$SOURCE_DIR"
echo "Created backup_${STAMP}.tar.gz"
"""


def build_template_corpus(target_samples: int, seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    rows: list[dict] = []
    language_buckets = [
        ("Python", PYTHON_TASKS, _python_template),
        ("JavaScript", JAVASCRIPT_TASKS, _javascript_template),
        ("Shell", SHELL_TASKS, _shell_template),
    ]

    while len(rows) < target_samples:
        language, tasks, renderer = rng.choice(language_buckets)
        task_name, prompt = rng.choice(tasks)
        rows.append(
            {
                "prompt": prompt,
                "text": renderer(task_name),
                "language": language,
                "generator": "template",
            }
        )
    return rows
