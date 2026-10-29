FROM golang:1.26.10-bookworm AS build
WORKDIR /src
COPY go.mod ./
COPY *.go ./
RUN CGO_ENABLED=0 GOFLAGS=-p=4 go build -trimpath -ldflags="-s -w" -o /out/ai-edge .

FROM gcr.io/distroless/static-debian12:nonroot
COPY --from=build /out/ai-edge /ai-edge
USER nonroot:nonroot
EXPOSE 8080
ENTRYPOINT ["/ai-edge"]
