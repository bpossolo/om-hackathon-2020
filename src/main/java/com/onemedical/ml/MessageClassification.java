package com.onemedical.ml;

/**
 * The result of classifying a patient message.
 */
public class MessageClassification {

  /**
   * The patient message.
   */
  public Message message;

  /**
   * The predictions for job role that should handle the patient's message.
   */
  public LabelProbability[] roles;

  /**
   * The predictions for message category.
   */
  public LabelProbability[] categories;
}
