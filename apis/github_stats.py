import requests

def get_github_repo_stats(owner, repo):
    url = f"https://api.github.com/repos/{owner}/{repo}"
    
    response = requests.get(url)
    
    if response.status_code != 200:
        print("Error fetching data")
        return
    
    data = response.json()

    stats = {
        "Repository": data["full_name"],
        "Stars": data["stargazers_count"],
        "Forks": data["forks_count"],
        "Watchers": data["watchers_count"],
        "Open Issues": data["open_issues_count"],
        "Size (KB)": data["size"],
        "Default Branch": data["default_branch"]
    }

    return stats


owner = "torvalds"
repo = "linux"

stats = get_github_repo_stats(owner, repo)

for key, value in stats.items():
    print(f"{key}: {value}")
