package org.cotton.gs;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.*;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.client.RestClient;

@RestController
public class PredictionController {
    private final RestClient client;

    public PredictionController(@Value("${python.api-url:http://localhost:8000}") String apiUrl) {
        this.client = RestClient.builder().baseUrl(apiUrl).build();
    }

    @GetMapping("/health")
    public String health() {
        return "Java cotton genomic selection service is running";
    }

    @PostMapping("/predict")
    public ResponseEntity<String> predict(@RequestBody PredictionRequest request) {
        String response = client.post()
            .uri("/predict")
            .contentType(MediaType.APPLICATION_JSON)
            .body(request)
            .retrieve()
            .body(String.class);
        return ResponseEntity.ok(response);
    }
}
