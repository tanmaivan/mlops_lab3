FROM fluent/fluentd:v1.16-debian-1

USER root

RUN gem install fluent-plugin-elasticsearch -v 5.5.0 --no-document \
    && gem sources --clear-all
USER fluent
