---
description: Common error messages and how to debug them
---

# Common errors

### Docker Issues

<details>

<summary>Docker Issues - storage full</summary>

* Storage full on submission box
  * there can be so many error patterns in shortage of storage

{% code title="Add error" overflow="wrap" %}
```
// Some code
```
{% endcode %}

In this case, a common problem is that multiple unused docker containers remain on the submission box. A simple solution is to prune unused containers:

```
docker system prune
```

or belows:

```
// Remove all stopped containers
% docker rm $(docker ps -a -q)
```

```
// or Remove all containers
% docker rm -f $(docker ps -a -q)
```

</details>

<details>

<summary>docker volume could not be changed even if the container was updated and relaunched</summary>

* Docker volume will not be changed and remained once it was created (except mount with -v). That is to say "exists independently".
* This is observed as well in using docker-compose. If you define and use volumes: in docker-compose.yml file, be cautious that the created volume will not be removed after invoking `docker compose down` .
* a workaround is to delete the docker volume beforehand

```
# Remove volumes
% docker volume rm <specified volume name>
# or 
% docker volume prune
```

</details>
