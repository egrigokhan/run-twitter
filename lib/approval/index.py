def approve(request, config):
    title = config.get("title", "Default Title")
    description = config.get("description", "Default Description")
    options = config.get("options", [])

    formatted_options = []

    for option in options:
        formatted_option = {
            "title": option.get("title", "Default Option Title"),
            "description": option.get("description", "Default Option Description"),
            "callback": option.get("callback", None)
        }
        formatted_options.append(formatted_option)

    return {
        "request": request,
        "title": title,
        "description": description,
        "options": formatted_options
    }